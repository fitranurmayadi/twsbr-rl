# Two-Wheeled Self-Balancing Robot (TWSBR) — Deep Reinforcement Learning & Dynamic Control

[![Simulation: PyBullet & MuJoCo](https://img.shields.io/badge/Physics-PyBullet%20%2F%20MuJoCo-blue.svg)]()
[![Framework: Gymnasium & Stable--Baselines3](https://img.shields.io/badge/RL-Gymnasium%20%2F%20SB3-green.svg)]()
[![Algorithms: PPO / SAC / TD3](https://img.shields.io/badge/Algorithms-PPO%20%7C%20SAC%20%7C%20TD3%20%7C%20LQR-purple.svg)]()
[![Model: ONNX Export](https://img.shields.io/badge/Inference-ONNX%20Runtime-orange.svg)]()

Comprehensive simulation, dynamic modeling, and reinforcement learning control benchmark for an inverted-pendulum **Two-Wheeled Self-Balancing Robot (TWSBR)**. Evaluates continuous action space deep RL policies against classical optimal state-feedback Linear Quadratic Regulators (LQR) and cascaded PID controllers under external impulse disturbances.

---

## 🦾 Core Highlights

- **Multi-Engine Physics Simulation**:
  - Custom Gym/Gymnasium environments built on **PyBullet 3D** and **MuJoCo** physics engines.
  - Realistic non-linear inverted pendulum kinematics with ground contact friction, motor torque saturation, and center-of-mass perturbations.
- **Deep Reinforcement Learning Benchmark Suite**:
  - Implements state-of-the-art continuous policy optimization algorithms:
    - **PPO** (Proximal Policy Optimization)
    - **SAC** (Soft Actor-Critic)
    - **TD3** (Twin Delayed DDPG)
    - **A2C** (Advantage Actor-Critic)
- **Baseline Comparative Controllers**:
  - Tuned full-state feedback **LQR** ( = -Kx$) based on linearized state-space matrices (, B, C, D$).
  - Cascaded multi-loop **PID** (Angle loop + Velocity loop).
  - Genetic Algorithm (GA) heuristic tuning for optimal PID gain selection.
- **Edge Deployment Ready**:
  - Trained PyTorch neural networks exported to **ONNX format** (model.onnx) for low-latency embedded inference.

---

## 📄 License

MIT License © [Fitra Nurmayadi](https://github.com/fitranurmayadi).