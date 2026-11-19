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
 *   - Any message on the reset topic (gz.msgs.Empty) teleports the drone back
 *     to its spawn pose, zeroes its velocity and disarms it.
 *
 * SDF parameters (all optional):
 *   <link_name>chassis</link_name>
 *   <topic>/model/<model_name>/cmd_vel</topic>
 *   <reset_topic>/model/<model_name>/reset</reset_topic>
 *   <cmd_timeout>0.5</cmd_timeout>
 *   <max_accel>20</max_accel>         per-axis command limit [m/s^2], force limit = mass x max_accel
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
#include <gz/msgs/empty.pb.h>
#include <gz/msgs/twist.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Gravity.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/Inertial.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/PoseCmd.hh>
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
    const double maxAccel = _sdf->Get<double>("max_accel", 20.0).first;
    const double wn = _sdf->Get<double>("attitude_wn", 10.0).first;
    this->attKp = wn * wn;
    this->attKd = 2.0 * wn;  // critically damped
    this->yawGain = _sdf->Get<double>("yaw_rate_gain", 10.0).first;
    this->linearDrag = _sdf->Get<double>("linear_drag", 1.0).first;
    const std::string topic = _sdf->Get<std::string>(
      "topic", "/model/" + model.Name(_ecm) + "/cmd_vel").first;
    const std::string resetTopic = _sdf->Get<std::string>(
      "reset_topic", "/model/" + model.Name(_ecm) + "/reset").first;

    this->modelEntity = _entity;

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
    this->maxForce = this->mass * maxAccel;

    const auto *gravity = _ecm.Component<gz::sim::components::Gravity>(gz::sim::worldEntity(_ecm));
    this->gravity = gravity ? gravity->Data() : gz::math::Vector3d(0, 0, -9.8);

    this->node.Subscribe(topic, &DroneForceController::OnCmd, this);
    this->node.Subscribe(resetTopic, &DroneForceController::OnReset, this);
    this->valid = true;

    gzmsg << "DroneForceController: link [" << linkName << "], mass " << this->mass
          << " kg, force limit " << this->maxForce << " N per axis, listening on ["
          << topic << "], reset on [" << resetTopic << "]" << std::endl;
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &_ecm) override
  {
    if (!this->valid || _info.paused)
      return;

    // Record the spawn pose the reset returns to. Not done in Configure: when a
    // model is spawned with the create service its pose is set after Configure.
    if (!this->spawnRecorded)
    {
      this->spawnPose = gz::sim::worldPose(this->modelEntity, _ecm);
      this->spawnRecorded = true;
      gzmsg << "DroneForceController: spawn pose [" << this->spawnPose << "]" << std::endl;
    }

    const double now = std::chrono::duration<double>(_info.simTime).count();

    // Velocity commands from a reset only need to act for one step
    if (this->clearVelocityCmds)
    {
      _ecm.RemoveComponent<gz::sim::components::LinearVelocityCmd>(this->link.Entity());
      _ecm.RemoveComponent<gz::sim::components::AngularVelocityCmd>(this->link.Entity());
      this->clearVelocityCmds = false;
    }

    gz::math::Vector3d forceCmd;
    double yawRateCmd = 0.0;
    bool reset = false;
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      reset = this->resetRequested;
      this->resetRequested = false;
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

    if (reset)
    {
      _ecm.SetComponentData<gz::sim::components::WorldPoseCmd>(this->modelEntity, this->spawnPose);
      this->link.SetLinearVelocity(_ecm, gz::math::Vector3d::Zero);
      this->link.SetAngularVelocity(_ecm, gz::math::Vector3d::Zero);
      this->clearVelocityCmds = true;
      this->armed = false;
      gzmsg << "DroneForceController: reset to spawn pose [" << this->spawnPose << "]" << std::endl;
      return;
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

  void OnReset(const gz::msgs::Empty &)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->resetRequested = true;
    // Drop the last command so the drone stays disarmed until a new up command
    this->cmdForce.Set(0, 0, 0);
    this->cmdYawRate = 0.0;
    this->lastCmdTime = -1e9;
    this->newCmd = false;
  }

  gz::transport::Node node;
  gz::sim::Link link;
  bool valid = false;
  bool armed = false;

  gz::sim::Entity modelEntity = gz::sim::kNullEntity;
  gz::math::Pose3d spawnPose;
  bool spawnRecorded = false;
  bool clearVelocityCmds = false;

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
  bool resetRequested = false;
};

}  // namespace drone_plugins

GZ_ADD_PLUGIN(drone_plugins::DroneForceController,
              gz::sim::System,
              drone_plugins::DroneForceController::ISystemConfigure,
              drone_plugins::DroneForceController::ISystemPreUpdate)
