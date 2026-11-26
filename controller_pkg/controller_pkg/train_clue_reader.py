#!/usr/bin/env python3
"""Train the clue banner character classifier (clue_reader.py).

Training data are banners drawn exactly like the competition's
clueboard_generator.py (clue_banner.png, UbuntuMono-R size 90, same text
positions) with random text, degraded to look like the drone's camera:
lighting, a slightly off crop, and the blur of a board 0.5 - 2 m away
(letters 5 - 40 px tall in the camera). Each banner is cut with the same
segmentation the reader uses, so the classifier learns from exactly the crops
it will see. Banners whose cut doesn't match the text (like the notebook's
"skip if the character count is wrong") are left out.

The default model is the team notebook's CNN, trained with PyTorch and
exported to ONNX (the drone runs it with OpenCV's dnn module, no PyTorch
needed). --model svm trains the older HOG + SVM instead.

Usage:  ros2 run controller_pkg train_clue_reader [--model cnn|svm] [--banners 3000] [--out PATH]
"""
import argparse
import os
import random
import string
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from controller_pkg import clue_reader as cr

FONT = '/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf'
BANNER = os.path.expanduser(
    '~/2026_competition_ws/src/2026_competition/src/world_pkg/world/clueboard_scripts/clue_banner.png')
ALPHABET = string.ascii_uppercase + string.digits


def render(base, font, clue_type, value):
    """Same drawing as clueboard_generator.banner_generator (BGR image)."""
    pil = Image.fromarray(base)
    draw = ImageDraw.Draw(pil)
    draw.text((cr.TYPE_X, 30), clue_type, (255, 0, 0), font=font)
    draw.text((cr.VALUE_X, 250), value, (255, 0, 0), font=font)
    return np.array(pil)


def random_text(max_len, min_len=1, spaces=True):
    n = random.randint(min_len, max_len)
    chars = [random.choice(ALPHABET) for _ in range(n)]
    if spaces and n >= 3:
        for _ in range(random.randint(0, 2)):
            i = random.randint(1, n - 2)
            if chars[i - 1] != ' ' and chars[i + 1] != ' ':
                chars[i] = ' '
    return ''.join(chars)


def degrade(banner, rng):
    """Make a rendered banner look like a flattened banner from the drone camera."""
    h, w = banner.shape[:2]
    image = banner.astype(np.float32)
    # Lighting: Gazebo shows the white banner as grey 95 - 210, plus a gentle gradient
    grey = rng.uniform(95, 210)
    gradient = np.linspace(rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1), w)[None, :, None]
    image = image * (grey / 255.0) * gradient
    # Off crop: small shift / scale / rotation / shear
    angle = rng.uniform(-2, 2)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[0, 0] *= rng.uniform(0.95, 1.05)
    m[1, 1] *= rng.uniform(0.93, 1.07)
    m[0, 1] += rng.uniform(-0.04, 0.04)
    m[0, 2] += rng.uniform(-15, 15)
    m[1, 2] += rng.uniform(-12, 12)
    image = cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # Distance: letters (63 px tall here) become 5 - 40 px in the camera
    letter = rng.uniform(5, 40)
    f = letter / 63.0
    small = cv2.resize(image, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        small = cv2.GaussianBlur(small, (0, 0), rng.uniform(0.3, 0.8))
    small = small + rng.normal(0, rng.uniform(1.5, 4.0), small.shape)  # camera noise
    image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(image, 0, 255).astype(np.uint8), letter


def expected_slots(text):
    """Slot pattern the segmentation should produce: None for spaces."""
    return [None if c == ' ' else c for c in text.strip()]


def make_dataset(n_banners, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    base = cv2.imread(BANNER)
    if base is None:
        raise FileNotFoundError(BANNER)
    font = ImageFont.truetype(FONT, 90)
    crops, labels, kept, letters = [], [], 0, []
    for _ in range(n_banners):
        clue_type = random.choice(cr.KEYWORDS) if random.random() < 0.6 else random_text(6, 3, False)
        value = random_text(cr.VALUE_MAX)
        banner, letter = degrade(render(base, font, clue_type, value), rng)
        ink, type_slots, value_slots = cr.segment(banner)
        ok = True
        pairs = []
        for slots, text in ((type_slots, clue_type), (value_slots, value)):
            want = expected_slots(text)
            if len(slots) != len(want) or any((s is None) != (c is None) for s, c in zip(slots, want)):
                ok = False
                break
            pairs += [(s, c) for s, c in zip(slots, want) if c is not None]
        if not ok:
            continue
        kept += 1
        letters.append(letter)
        for box, char in pairs:
            crops.append(cr.crop_character(ink, box))
            labels.append(cr.CHARACTERS.index(char))
    return np.array(crops, np.float32), np.array(labels, np.int32), kept, letters


def hog(crops):
    return np.stack([cr.hog_features(c) for c in crops]).astype(np.float32)


def train_svm(x, y):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(10.0)
    svm.setGamma(0.5)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 2000, 1e-4))
    svm.train(x, cv2.ml.ROW_SAMPLE, y)
    return svm


def train_cnn(x, y, xv, yv, out, epochs=25, seed=0):
    """The team notebook's CNN (Conv32-Pool-Conv64-Pool-Conv64-Dense128-Dropout),
    trained with PyTorch on blueness crops and exported to ONNX for cv2.dnn.
    Training needs torch and onnx (pip install torch onnx); the drone only needs OpenCV."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, len(cr.CHARACTERS)))
    xt = torch.from_numpy(x[:, None])
    yt = torch.from_numpy(y.astype(np.int64))
    xvt = torch.from_numpy(xv[:, None])
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(yt))
        total = 0.0
        for i in range(0, len(order), 128):
            batch = order[i:i + 128]
            xb = xt[batch]
            # Small random shifts: the crop box is never exactly centred
            dx, dy = np.random.randint(-2, 3, 2)
            xb = torch.roll(xb, shifts=(int(dy), int(dx)), dims=(2, 3))
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yt[batch])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        schedule.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(xvt).argmax(1).numpy() == yv).mean())
        print(f'  epoch {epoch + 1:2d}: loss {total / len(yt):.4f}, validation accuracy {acc:.4f}')

    model.eval()
    exported = nn.Sequential(model, nn.Softmax(dim=1))
    torch.onnx.export(exported, torch.zeros(1, 1, cr.CROP_SIZE, cr.CROP_SIZE), out,
                      input_names=['crop'], output_names=['probs'],
                      dynamic_axes={'crop': {0: 'n'}, 'probs': {0: 'n'}}, opset_version=13,
                      dynamo=False)
    return acc


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--banners', type=int, default=3000)
    parser.add_argument('--model', choices=['cnn', 'svm'], default='cnn')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models',
                                'clue_chars_cnn.onnx' if args.model == 'cnn' else 'clue_chars_svm.yml')

    t = time.time()
    x, y, kept, letters = make_dataset(args.banners, seed=1)
    print(f'{kept}/{args.banners} training banners cut cleanly -> {len(y)} characters '
          f'({time.time() - t:.0f} s)')
    if kept:
        small = np.mean(np.array(letters) < 8)
        print(f'  letter heights 5 - 40 px; {small:.0%} of the kept banners have letters under 8 px')
    xv, yv, kept_v, _ = make_dataset(max(300, args.banners // 6), seed=2)
    print(f'{kept_v} validation banners -> {len(yv)} characters')

    t = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.model == 'cnn':
        acc = train_cnn(x, y, xv, yv, args.out, args.epochs)
        print(f'CNN: validation character accuracy {acc:.4f} (trained in {time.time() - t:.0f} s)')
    else:
        svm = train_svm(hog(x), y)
        _, pred = svm.predict(hog(xv))
        acc = float(np.mean(pred.ravel().astype(int) == yv))
        print(f'SVM: validation character accuracy {acc:.4f} (trained in {time.time() - t:.0f} s)')
        svm.save(args.out)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
