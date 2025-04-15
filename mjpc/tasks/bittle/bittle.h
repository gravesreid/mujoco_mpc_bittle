// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_TASKS_BITTLE_BITTLE_H_
#define MJPC_TASKS_BITTLE_BITTLE_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {

class BittleFlat : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const BittleFlat* task)
        : mjpc::BaseResidualFn(task) {}
    ResidualFn(const ResidualFn&) = default;
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class BittleFlat;
    //  ============  enums  ============
    // modes
    enum BittleMode {
      kModeQuadruped = 0,
      kModeWalk,
      kModeScramble,
      kModeFlip,
      kNumMode
    };

    // feet
    enum BittleFoot {
      kFootLF  = 0,  // left-front
      kFootLB,       // left-back
      kFootRF,       // right-front
      kFootRB,       // right-back
      kNumFoot
    };

    // gaits
    enum BittleGait {
      kGaitStand = 0,
      kGaitWalk,
      kGaitTrot,
      kGaitCanter,
      kGaitGallop,
      kNumGait
    };

    //  ============  constants  ============
    constexpr static BittleFoot kFootAll[kNumFoot] = {kFootLF, kFootLB,
                                                  kFootRF, kFootRB};
    constexpr static BittleFoot kFootHind[2] = {kFootLB, kFootRB};
    constexpr static BittleGait kGaitAll[kNumGait] = {kGaitStand, kGaitWalk,
                                                  kGaitTrot, kGaitCanter,
                                                  kGaitGallop};

    // gait phase signature (normalized)
    constexpr static double kGaitPhase[kNumGait][kNumFoot] =
    {
    // LF     LB     RF     RB
      {0,     0,     0,     0   },   // stand
      {0,     0.75,  0.5,   0.25},   // walk
      {0,     0.75,   0.5,   0.25   },   // trot
      {0,     0.33,  0.33,  0.66},   // canter
      {0,     0.4,   0.05,  0.35}    // gallop
    };

    // gait parameters, set when switching into gait
    constexpr static double kGaitParam[kNumGait][6] =
    {
    // duty ratio  cadence  amplitude  balance   upright   height
    // unitless    Hz       meter      unitless  unitless  unitless
      {1,          1,       0,         0,        1,        1},      // stand
      {0.96,       1,       0.01,      0.2,        1,        1},      // walk
      {0.9,       2,       0.03,      0.2,      1,        1},      // trot
      {0.9,        4,       0.05,      0.03,     0.5,      0.2},    // canter
      {0.9,        3.5,     0.10,      0.03,     0.2,      0.1}     // gallop
    };

    // velocity ranges for automatic gait switching, meter/second
    constexpr static double kGaitAuto[kNumGait] =
    {
      0,     // stand
      0.01,  // walk
      0.02,  // trot
      0.6,   // canter
      2,     // gallop
    };

    // automatic gait switching: time constant for com speed filter
    constexpr static double kAutoGaitFilter = 0.2;    // second

    // automatic gait switching: minimum time between switches
    constexpr static double kAutoGaitMinTime = 1;     // second

    // target torso height over feet when quadrupedal
    constexpr static double kHeightQuadruped = .8;  // meter - adjusted for Bittle

    // radius of foot geoms
    constexpr static double kFootRadius = 0.05;       // meter - adjusted for Bittle

    // below this target yaw velocity, walk straight
    constexpr static double kMinAngvel = 0.01;        // radian/second

    // flip: crouching height, from which leap is initiated
    constexpr static double kCrouchHeight = 0.8;     // meter - adjusted for Bittle

    // flip: leap height, beginning of flight phase
    constexpr static double kLeapHeight = 0.3;        // meter - adjusted for Bittle

    // flip: maximum height of flight phase
    constexpr static double kMaxHeight = 0.95;         // meter - adjusted for Bittle

    //  ============  methods  ============
    // return internal phase clock
    double GetPhase(double time) const;

    // return current gait
    BittleGait GetGait() const;

    // compute average foot position, depending on mode
    void AverageFootPos(double avg_foot_pos[3],
                        double* foot_pos[kNumFoot]) const;

    // return normalized target step height
    double StepHeight(double time, double footphase, double duty_ratio) const;

    // compute target step height for all feet
    void FootStep(double step[kNumFoot], double time, BittleGait gait) const;

    // walk horizontal position given time
    void Walk(double pos[2], double time) const;

    // height during flip
    double FlipHeight(double time) const;

    // orientation during flip
    void FlipQuat(double quat[4], double time) const;

    //  ============  task state variables, managed by Transition  ============
    BittleMode current_mode_     = kModeQuadruped;
    double last_transition_time_ = -1;

    // common mode states
    double mode_start_time_  = 0;
    double position_[3]       = {0};

    // walk states
    double heading_[2]        = {0};
    double speed_             = 0;
    double angvel_            = 0;

    // backflip states
    double ground_            = 0;
    double orientation_[4]    = {0};
    double save_gait_switch_  = 0;
    std::vector<double> save_weight_;

    // gait-related states
    double current_gait_      = kGaitStand;
    double phase_start_       = 0;
    double phase_start_time_  = 0;
    double phase_velocity_    = 0;
    double com_vel_[2]        = {0};
    double gait_switch_time_  = 0;

    //  ============  constants, computed in Reset()  ============
    int torso_body_id_        = -1;
    int head_site_id_         = -1;
    int goal_mocap_id_        = -1;
    int gait_param_id_        = -1;
    int gait_switch_param_id_ = -1;
    int flip_dir_param_id_    = -1;
    int cadence_param_id_     = -1;
    int amplitude_param_id_   = -1;
    int duty_param_id_        = -1;
    int upright_cost_id_      = -1;
    int balance_cost_id_      = -1;
    int height_cost_id_       = -1;
    int foot_geom_id_[kNumFoot];
    
    // derived kinematic quantities describing flip trajectory
    double gravity_           = 0;
    double jump_vel_          = 0;
    double flight_time_       = 0;
    double jump_acc_          = 0;
    double crouch_time_       = 0;
    double leap_time_         = 0;
    double jump_time_         = 0;
    double crouch_vel_        = 0;
    double land_time_         = 0;
    double land_acc_          = 0;
    double flight_rot_vel_    = 0;
    double jump_rot_vel_      = 0;
    double jump_rot_acc_      = 0;
    double land_rot_acc_      = 0;
  };

  BittleFlat() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

  // call base-class Reset, save task-related ids
  void ResetLocked(const mjModel* model) override;

  // draw task-related geometry in the scene
  void ModifyScene(const mjModel* model, const mjData* data,
                   mjvScene* scene) const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(residual_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  friend class ResidualFn;
  ResidualFn residual_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_BITTLE_BITTLE_H_