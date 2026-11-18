/*
 * DroneForceController
 *
 * Gazebo system plugin that flies a single rigid-body drone with force
 * commands, as the competition rules require for flying robots:
 *   - Linear motion is force controlled. The gz Twist received on the command
 *     topic is interpreted as:
 *       linear.x  forward force  [N]  (yaw-aligned body frame)
 *       linear.y  left force     [N]  (yaw-aligned body frame)
 *       linear.z  up force       [N]  (on top of gravity compensation)
 *       angular.z yaw rate       [rad/s]
 *   - Gravity compensation (m * g) is applied once the drone is "armed",
 *     i.e. after the first positive linear.z command. Before that the drone
 *     just sits on the ground.
 *   - A linear drag force (-linear_drag * velocity) models air resistance so
 *     the drone slows down when no force is commanded.
 *   - Roll and pitch are held level with a PD torque controller.
 *   - If no command is received for <cmd_timeout> seconds the force command
 *     is zeroed, so the drone hovers (it stays armed).
 *
 * SDF parameters (all optional):
 *   <link_name>chassis</link_name>
 *   <topic>/model/<model_name>/cmd_vel</topic>
 *   <cmd_timeout>0.5</cmd_timeout>
 *   <max_force>20</max_force>
 *   <linear_drag>1.0</linear_drag>    N per m/s
 *   <attitude_wn>10</attitude_wn>     natural frequency of roll/pitch hold
 *   <yaw_rate_gain>10</yaw_rate_gain>
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>

#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/twist.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Gravity.hh>
#include <gz/sim/components/Inertial.hh>
#include <gz/transport/Node.hh>

namespace drone_plugins
{

class DroneForceController
  : public gz::sim::System,
    public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &_ecm,
                 gz::sim::EventManager &) override
  {
    gz::sim::Model model(_entity);
    if (!model.Valid(_ecm))
    {
      gzerr << "DroneForceController must be attached to a model." << std::endl;
      return;
    }

    const std::string linkName = _sdf->Get<std::string>("link_name", "chassis").first;
    this->cmdTimeout = _sdf->Get<double>("cmd_timeout", 0.5).first;
    this->maxForce = _sdf->Get<double>("max_force", 20.0).first;
    const double wn = _sdf->Get<double>("attitude_wn", 10.0).first;
    this->attKp = wn * wn;
    this->attKd = 2.0 * wn;  // critically damped
    this->yawGain = _sdf->Get<double>("yaw_rate_gain", 10.0).first;
    this->linearDrag = _sdf->Get<double>("linear_drag", 1.0).first;
    const std::string topic = _sdf->Get<std::string>(
      "topic", "/model/" + model.Name(_ecm) + "/cmd_vel").first;

    this->link = gz::sim::Link(model.LinkByName(_ecm, linkName));
    if (!this->link.Valid(_ecm))
    {
      gzerr << "DroneForceController: link [" << linkName << "] not found." << std::endl;
      return;
    }
    this->link.EnableVelocityChecks(_ecm, true);

    // Mass and inertia of the link (fixed-joint children are lumped into it)
    const auto *inertial = _ecm.Component<gz::sim::components::Inertial>(this->link.Entity());
    this->mass = inertial->Data().MassMatrix().Mass();
    this->moi = inertial->Data().MassMatrix().DiagonalMoments();

    const auto *gravity = _ecm.Component<gz::sim::components::Gravity>(gz::sim::worldEntity(_ecm));
    this->gravity = gravity ? gravity->Data() : gz::math::Vector3d(0, 0, -9.8);

    this->node.Subscribe(topic, &DroneForceController::OnCmd, this);
    this->valid = true;

    gzmsg << "DroneForceController: link [" << linkName << "], mass " << this->mass
          << " kg, listening on [" << topic << "]" << std::endl;
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &_ecm) override
  {
    if (!this->valid || _info.paused)
      return;

    const double now = std::chrono::duration<double>(_info.simTime).count();

    gz::math::Vector3d forceCmd;
    double yawRateCmd = 0.0;
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      if (this->newCmd)
      {
        this->lastCmdTime = now;
        this->newCmd = false;
      }
      if (now - this->lastCmdTime <= this->cmdTimeout)
      {
        forceCmd = this->cmdForce;
        yawRateCmd = this->cmdYawRate;
      }
    }

    if (forceCmd.Z() > 0.0)
      this->armed = true;
    if (!this->armed)
      return;

    const auto pose = this->link.WorldPose(_ecm);
    const auto angVelWorld = this->link.WorldAngularVelocity(_ecm);
    const auto linVelWorld = this->link.WorldLinearVelocity(_ecm);
    if (!pose || !angVelWorld || !linVelWorld)
      return;
    const gz::math::Quaterniond rot = pose->Rot();

    // Linear: commanded force in the yaw-aligned frame + gravity compensation + drag
    for (int i = 0; i < 3; ++i)
      forceCmd[i] = std::clamp(forceCmd[i], -this->maxForce, this->maxForce);
    const gz::math::Quaterniond yawOnly(0.0, 0.0, rot.Yaw());
    const gz::math::Vector3d force = yawOnly.RotateVector(forceCmd)
      - this->mass * this->gravity - this->linearDrag * (*linVelWorld);

    // Angular: PD to hold roll/pitch level, P on yaw rate
    const gz::math::Vector3d w = rot.RotateVectorReverse(*angVelWorld);
    const gz::math::Vector3d alpha(
      -this->attKp * rot.Roll() - this->attKd * w.X(),
      -this->attKp * rot.Pitch() - this->attKd * w.Y(),
      this->yawGain * (yawRateCmd - w.Z()));
    const gz::math::Vector3d torqueBody(
      this->moi.X() * alpha.X(), this->moi.Y() * alpha.Y(), this->moi.Z() * alpha.Z());

    this->link.AddWorldForce(_ecm, force);
    this->link.AddWorldWrench(_ecm, gz::math::Vector3d::Zero, rot.RotateVector(torqueBody));
  }

private:
  void OnCmd(const gz::msgs::Twist &_msg)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->cmdForce.Set(_msg.linear().x(), _msg.linear().y(), _msg.linear().z());
    this->cmdYawRate = _msg.angular().z();
    this->newCmd = true;
  }

  gz::transport::Node node;
  gz::sim::Link link;
  bool valid = false;
  bool armed = false;

  double mass = 1.0;
  gz::math::Vector3d moi{1, 1, 1};
  gz::math::Vector3d gravity{0, 0, -9.8};

  double cmdTimeout = 0.5;
  double maxForce = 20.0;
  double attKp = 100.0;
  double attKd = 20.0;
  double yawGain = 10.0;
  double linearDrag = 1.0;

  std::mutex mutex;
  gz::math::Vector3d cmdForce;
  double cmdYawRate = 0.0;
  bool newCmd = false;
  double lastCmdTime = -1e9;
};

}  // namespace drone_plugins

GZ_ADD_PLUGIN(drone_plugins::DroneForceController,
              gz::sim::System,
              drone_plugins::DroneForceController::ISystemConfigure,
              drone_plugins::DroneForceController::ISystemPreUpdate)
