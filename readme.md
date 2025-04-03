# Clash Royale AI Bot (M1 Mac / BlueStacks)

## Project Goal

This project aims to develop an AI agent capable of playing the mobile game Clash Royale. The game will run within the BlueStacks Android emulator (optimized version for Apple Silicon) on an M1/M2 Mac. The project serves as an exploration of computer vision, reinforcement learning, and system automation techniques applied to a real-time strategy game.

## Core Technologies

*   **Platform:** macOS (Apple Silicon M1/M2)
*   **Emulator:** BlueStacks (M1-compatible version)
*   **Game:** Clash Royale
*   **Programming Language:** Python 3
*   **Core Libraries:**
    *   **Computer Vision:** OpenCV (`opencv-python`)
    *   **Machine Learning:** PyTorch or TensorFlow/Keras (for CNN, RNN/LSTM/Transformer, RL)
    *   **Screen Capture:** macOS `screencapture` utility via `subprocess`, potentially `mss`
    *   **Window Interaction:** AppleScript via `subprocess` (for window geometry)
    *   **Action Emulation:** TBD (Potentially `pyautogui`, ADB via BlueStacks, or other macOS automation tools)
    *   **Numerical Computation:** NumPy

## Project Plan / Workflow

The AI agent will operate in a loop, performing the following steps at each decision point:

1.  **Environment Check:** Ensure BlueStacks and Clash Royale are running.
2.  **Window Capture:**
    *   Identify the geometry (position and size) of the "BlueStacks Air" window using AppleScript.
    *   Capture the specific region of the screen corresponding to the BlueStacks window using `screencapture -R`.
3.  **State Extraction (Parallel Processing):**
    *   **Structured Data (OpenCV):** Process the captured screenshot using OpenCV functions to extract:
        *   Current Elixir count
        *   Player Tower Health (King, Princess Towers)
        *   Opponent Tower Health
        *   Time remaining
        *   Cards currently in hand (potentially identifying type and cost)
        *   *Output:* A numerical vector representing this structured data.
    *   **Visual Features (CNN):** Feed the main arena portion of the screenshot into a pre-trained or concurrently trained Convolutional Neural Network (CNN).
        *   *Output:* A feature vector/tensor representing troop positions, types, spell effects, and other visual game elements.
4.  **State Combination:** Concatenate or otherwise combine the structured data vector (from OpenCV) and the visual feature vector (from CNN) into a single state representation for the current time step.
5.  **Temporal Processing (RNN/LSTM/Transformer):** Maintain a sequence of recent combined states and feed them into a recurrent layer (like LSTM or potentially a Transformer) to capture the game's dynamics and history.
    *   *Output:* A context-aware state representation encoding recent game events.
6.  **Action Masking:** Based on the current game state (especially elixir count from OpenCV and known card costs):
    *   Generate a mask indicating which cards in the hand are currently playable (affordable).
    *   Generate a mask indicating valid placement locations on the arena grid for the potential actions (considering troop vs. spell vs. building, placement rules).
7.  **Decision Making (RL Policy Network):**
    *   Input the context-aware state representation into the Reinforcement Learning agent's policy network head.
    *   The network outputs probabilities/logits for actions (e.g., selecting a card slot, selecting placement coordinates/heatmap).
    *   Apply the action masks (from step 6) to invalidate illegal moves (e.g., set logits of invalid actions to -infinity before softmax or sampling).
    *   Select the final action (which card to play and where).
8.  **Action Execution:** Translate the selected action into commands sent to the BlueStacks emulator to simulate the necessary clicks and drags. This requires a reliable method for input emulation.
9.  **Loop:** Repeat steps 2-8 for the next game step/decision point.

## Reinforcement Learning Aspect

*   **Algorithm:** TBD (e.g., DQN, PPO, A3C variants suitable for the action space).
*   **State:** Defined by the combined output of OpenCV, CNN, and RNN/LSTM layers.
*   **Actions:** Discrete (card selection) and potentially continuous or discretized grid (placement). Action masking is crucial.
*   **Reward Function:** Needs careful design (reward shaping). Potential components:
    *   Positive reward for damaging opponent towers.
    *   Large positive reward for destroying towers.
    *   Positive reward for destroying opponent troops (elixir advantage).
    *   Negative reward for taking tower damage.
    *   Large negative reward for losing towers.
    *   Negative reward for wasted elixir / inefficient plays.
    *   Large reward/penalty for winning/losing the game.

## Key Challenges

*   **Real-time Performance:** The entire capture -> process -> decide -> act loop must be fast enough for real-time gameplay.
*   **Robust Computer Vision:** Handling variations in arenas, troop levels/skins, new cards, and potential UI changes.
*   **Complex RL Training:** Large state space, complex action space (card + placement), delayed rewards, reward shaping, exploration vs. exploitation. Requires significant training time and computational resources.
*   **Reliable Action Emulation:** Accurately and reliably simulating clicks/drags within BlueStacks can be tricky.
*   **Emulator Stability:** Ensuring BlueStacks runs consistently without performance degradation.
*   **Game Updates:** The AI may need retraining or adjustments if Supercell significantly changes the game UI or mechanics.

## Setup (Placeholder)

1.  Install Python 3.
2.  Install BlueStacks (M1-compatible version).
3.  Install Clash Royale within BlueStacks.
4.  Set up a Python virtual environment (`venv` or `conda`).
5.  Install required Python packages (`pip install -r requirements.txt`).
6.  Grant macOS Permissions:
    *   **Accessibility:** For the script runner (Terminal/IDE) to get window geometry via AppleScript.
    *   **Screen Recording:** For the script runner to capture the screen via `screencapture`.

## Usage (Placeholder)

```bash
# Example command to start training or running the agent
python main_agent.py --mode train --load_model path/to/model.pth