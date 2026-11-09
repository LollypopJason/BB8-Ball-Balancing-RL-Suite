# BB-8 real-world verification checklist

Quick experiments that compare the real robot with the simulator. Do them in order: each one assumes the previous ones passed. Runner commands are run on the Jetson in the deployment package.

## 1. Measure & enter your robot
*Measure - about 10 min.* Every other prediction depends on these numbers.

**Equipment:** kitchen / luggage scale; tape measure or calliper; ruler + knife edge (or a pencil)

1. Weigh the bare ball and the complete head (with battery, Jetson, motors, wheels).
2. Measure the ball diameter at 3 places and average.
3. Head CoG: lay the fully assembled head on its side across a pencil and roll it until it balances; measure the balance point's height above the head plate, then add the gap between the ball top and the head plate (measured on the assembled robot) = CoG height above the ball top (+-1 cm is fine).
4. Measure the wheel diameter and the contact (zenith) angle from your CAD / with a protractor.

| Measure | Unit | Simulator prediction | Your value |
|---|---|---|---|
| Ball diameter | m | 0.5 | |
| Ball mass | kg | 2.5 | |
| Head mass | kg | 2.2 | |
| Head CoG above ball top | m | 0.085 | |
| Wheel diameter | m | 0.058 | |
| Zenith contact angle | deg | 31 | |

In the app: *Apply measured values to the configuration*.

## 2. Encoder count check (by hand)
*Quick check - about 2 min.* Encoder velocity is a policy input; a wrong sign makes every learned controller fail instantly.

**Equipment:** robot on a stand (ball removed); a tape mark on each wheel

1. Power the Jetson + encoders (motors may stay unpowered).
2. Run the runner test, then turn each wheel exactly 10 turns by hand in its positive direction.
3. Wrong count -> magnet gap / mux channel; negative -> flip encoder_direction_sign.

Runner: `python3 jetson_control_loop.py --test encoder-turns`

| Measure | Unit | Simulator prediction | Your value |
|---|---|---|---|
| Wheel 1 turns | rev | 10 | |
| Wheel 2 turns | rev | 10 | |
| Wheel 3 turns | rev | 10 | |

## 3. Wiring & sign check (interactive)
*Quick check - about 3 min.* The simulator's sign conventions must match the hardware exactly.

**Equipment:** robot on a stand (ball removed); the Jetson terminal

1. Run the runner test and follow the prompts: tilt nose-down, tilt right side down, rotate the head counter-clockwise, turn each wheel forward by hand.
2. Every value must be True. Fix IMU axes in the Arduino sketch (kAxisMap / kAxisSign) and encoder signs in robot_hw_config.json, then repeat.

Runner: `python3 jetson_control_loop.py --test signs`

| Measure | Unit | Simulator prediction | Your value |
|---|---|---|---|
| Pitch + when nose down |  | 1 | |
| Roll + when right side down |  | 1 | |
| Yaw rate + when turning CCW |  | 1 | |
| Encoder 1 + when turned forward |  | 1 | |
| Encoder 2 + when turned forward |  | 1 | |
| Encoder 3 + when turned forward |  | 1 | |

## 4. IMU noise & drift (robot still)
*Quick check - about 2 min.* Sim-to-real: train with the noise your sensor really has.

**Equipment:** robot standing still on its stand or held still

1. Run the runner test (60 s), do not touch the robot.
2. Import the JSON. 'Apply' writes your measured noise into the IMU randomisation ranges.

Runner: `python3 jetson_control_loop.py --test imu-noise`

| Measure | Unit | Simulator prediction | Your value |
|---|---|---|---|
| Roll noise std | deg | 0.2292 | |
| Gyro noise std | rad/s | 0.0125 | |
| Accel noise std | m/s^2 | 0.07 | |
| Yaw drift | deg/min | 0.6657 | |
| Accel magnitude (gravity) | m/s^2 | 9.81 | |

In the app: *Use measured IMU noise for domain randomisation*.

## 5. Ball roll-out on a ramp (no electronics)
*Quick check - about 5 min.* Rolling resistance of a hand-coated ball on your floor is hard to guess - and matters for drift.

**Equipment:** the bare ball; a straight board 1.0 m long, one end raised 2 cm (books); tape measure; a long flat hallway (the ball rolls several metres)

1. Raise one end of the board by exactly 2 cm.
2. Place the ball at the top end and release it without pushing.
3. Measure from the foot of the ramp to where the ball stops. Repeat 3x, enter the average.
4. 'Apply' identifies the ball's rolling resistance on your floor.

| Measure | Unit | Simulator prediction | Your value |
|---|---|---|---|
| Roll-out distance | m | 5.868 | |

In the app: *Identify rolling resistance from the roll-out*.

## 6. Wheel -> ball transmission ratio (rig)
*Quick check - about 5 min.* Checks wheel diameter, microstep setting and roller slip in one go.

**Equipment:** validation rig: one motor + omniwheel pressing on the ball, ball free to turn (e.g. on 3 casters); tape mark on the ball

1. Run the runner test (motor 1 turns the wheel 10 revolutions at 10 rad/s).
2. Count the ball revolutions about the wheel axle (fractions by the mark position).

Runner: `python3 jetson_control_loop.py --test ratio`

| Measure | Unit | Simulator prediction | Your value |
|---|---|---|---|
| Ball revolutions | rev | 1.145 | |

## 7. Motor step response
*Quick check - about 3 min.* Speed ratio != 1 -> microsteps / missed steps; slow response -> latency the policy must handle.

**Equipment:** robot on a stand, wheels spinning freely

1. Run the runner test: each motor +10 rad/s, stop, -10 rad/s.
2. Import the JSON (motor 1 is compared with the simulated stepper).

Runner: `python3 jetson_control_loop.py --test motor-step`

| Measure | Unit | Simulator prediction | Your value |
|---|---|---|---|
| Motor 1 time to 63 % | s | 0.03 | |
| Motor 1 speed ratio |  | 0.998 | |

## 8. Spin-in-place (Jacobian & signs)
*Quick check - about 2 min.* Validates zenith angle, wheel radius, motor order and direction signs - exactly the Jacobian.

**Equipment:** robot on the floor, someone holding the head upright

1. Hold the head level with both hands (ball on the floor).
2. Run the runner test: all wheels +10 rad/s for 5 s -> the head rotates about the vertical axis.
3. Compare the yaw rate (sign AND magnitude).

Runner: `python3 jetson_control_loop.py --test spin`

> Safety: Keep fingers away from the omniwheels.

| Measure | Unit | Simulator prediction | Your value |
|---|---|---|---|
| Head yaw rate | rad/s | 2.252 | |

## 9. Drive straight (head held)
*Quick check - about 3 min.* If the robot moves sideways or backwards, the policy's 'forward' differs from the hardware's.

**Equipment:** robot on the floor, someone holding the head upright; tape measure, floor tape at the start

1. Point the eye along the floor tape.
2. Run the runner test (0.2 m/s for 3 s, head held level).
3. Measure distance and direction of travel relative to the eye direction.

Runner: `python3 jetson_control_loop.py --test straight`

| Measure | Unit | Simulator prediction | Your value |
|---|---|---|---|
| Distance travelled | m | 0.6 | |
| Direction vs. eye | deg | 0 | |

## 10. Passive fall time (motors holding)
*Measure - about 5 min.* Checks the passive (open-loop) dynamics: masses, inertias and that the steppers really hold.

**Equipment:** the complete robot; a helper to catch it; soft mat

1. Motors powered and holding (zero speed).
2. Balance the robot by hand, tilt ~3 deg, release, and catch it after ~30 deg.
3. The runner test records the IMU; or film at 120 fps and count frames from 5 to 25 deg.
4. Compare: a big mismatch points at wrong masses / inertia / wheel holding torque (the fall time is only weakly sensitive to the CoG height, so use example 1 for the CoG).

Runner: `python3 jetson_control_loop.py --test fall-time`

> Safety: Always catch the robot; never let a 5+ kg robot fall freely.

| Measure | Unit | Simulator prediction | Your value |
|---|---|---|---|
| Time from 5 to 25 deg | s | 0.64 | |

## 11. First balance with the classical controller
*Closed loop - about 5 min.* A known-good baseline separates hardware problems from RL problems.

**Equipment:** complete robot; tether rope from the ceiling / a frame; kill switch on the motor supply

1. Run: jetson_control_loop.py --controller classical --seconds 30 (with tether).
2. Read max tilt and odometry drift from the summary / log.
3. Only after this works: try the RL policy.

> Safety: Tether + kill switch. Stay clear of the head height.

| Measure | Unit | Simulator prediction | Your value |
|---|---|---|---|
| Balanced for | s | 30 | |
| Max tilt | deg | 1.613 | |
| Drift after 30 s | m | 0.06264 | |

## 12. Push recovery (10 N, 0.2 s)
*Closed loop - about 5 min.* Compares disturbance rejection of sim and robot - tells you if the randomisation ranges are wide enough.

**Equipment:** balancing robot (classical or RL); luggage / spring scale; tether

1. Hook the spring scale to the head, pull horizontally to ~10 N (1 kg on the scale) for ~0.2 s and release.
2. Read the maximum tilt and the time until the tilt is back below 2 deg (from the log).

> Safety: Tether. Start with ~5 N.

| Measure | Unit | Simulator prediction | Your value |
|---|---|---|---|
| Max tilt | deg | 8.175 | |
| Recovery time | s | 2.53 | |

