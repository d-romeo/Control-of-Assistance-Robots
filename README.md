# 🤖 Assistive Robotic Arm – Voice and Vision-Guided Simulation

This repository documents the development and simulation of an **assistive robotic system** designed to support individuals with limited mobility through **voice commands** and **computer vision**.

## 🧠 Context and Motivation

In the past decade, assistive robotics has emerged as a multidisciplinary field focused on improving the quality of life of individuals with physical limitations. These systems aim to integrate seamlessly into everyday and clinical environments to **enhance user autonomy** and reduce dependence on third-party care.

This project addresses a common scenario: a person, due to an accident, surgery, or other debilitating condition, is confined to a bed and **unable to perform basic tasks** like picking up utensils, a glass of water, or essential medication. Our goal is to simulate an intelligent robotic manipulator capable of **handing objects to bedridden patients**, entirely guided by **natural voice commands**.

## 🎯 Project Objectives

- Design a robot-assisted scenario for **daily-life object retrieval**.
- Implement **speech recognition** to allow intuitive human-robot interaction.
- Use **computer vision** to identify and localize objects in the environment.
- Simulate all components using **CoppeliaSim**, with full control logic implemented through custom code.

## 🧩 System Architecture

The project is structured into the following modules:

- 🎙️ **Voice Command Interface**: Captures and interprets patient voice instructions.
- 🧍‍♂️ **User Intention Processing**: Converts speech input into actionable robot tasks.
- 🧠 **Object Recognition**: Identifies objects like cups, cutlery, or pill boxes using vision algorithms.
- 🤖 **Robot Control Module**: Plans and executes the motion to pick up and deliver the requested item.
- 🧪 **Simulation Environment**: All logic is tested and validated in a **virtual simulation** to ensure safety and flexibility during design.

## 🛠️ Technologies Used

- **CoppeliaSim (V-REP)** for 3D simulation
- **Python** for high-level logic
- **Custom libraries** for:
  - Speech recognition
  - Object detection
  - Robot control interface

## 🎓 Educational Value

This project was conducted as part of a **Master’s Degree in Industrial Automation**, in collaboration with the **Intelligent Robotics Lab**. It represents an **advanced academic project** combining robotics, artificial intelligence, and user-centered design in healthcare.

> The simulation serves as a **foundational step** for the development of a real physical prototype in the future.


## 📁 Repository Structure

```bash
📦Assistive-Robot
 ┣ 📂src
 ┃ ┣ 📜voice_control.py
 ┃ ┣ 📜object_detection.py
 ┃ ┗ 📜robot_controller.py
 ┣ 📂sim_model
 ┃ ┗ 📜CoppeliaSim_scene.ttt
 ┣ 📜README.md
 ┗ 📜requirements.txt

