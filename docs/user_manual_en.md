# User Manual - SISTEMA_EMG

## Introduction

Welcome to SISTEMA_EMG, an advanced platform for acquisition, processing, and classification of electromyographic (EMG) signals with direct application in myoelectric prosthesis control. This manual provides detailed instructions on how to install, configure, and use the system in its different operation modes.

## Installation

### System Requirements

**Hardware:**
- Arduino Uno/Mega/Nano
- MyoWare 2.0 sensor
- INMOVE prosthesis (optional for physical mode)
- Computer with USB port

**Software:**
- Python 3.8 or higher
- Python libraries (automatically installed via requirements.txt)
- Operating system: Windows 10/11, macOS, Linux

### Installation Procedure

1. **Clone the repository or extract the ZIP file:**
   ```
   git clone https://github.com/your-username/SISTEMA_EMG.git
   ```

2. **Navigate to the project directory:**
   ```
   cd SISTEMA_EMG
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Connect the hardware (if using physical mode):**
   - Connect the Arduino to the computer via USB
   - Connect the MyoWare 2.0 sensor to the Arduino according to the diagram below:
     - MyoWare VCC pin → Arduino 5V
     - MyoWare GND pin → Arduino GND
     - MyoWare SIG pin → Arduino analog pin A0
   - Position the electrodes on the target muscle following the instructions in the "Electrode Positioning" section

## Starting the System

To start SISTEMA_EMG, run the following command in the terminal:

```
python main.py
```

By default, the system will start in simulated mode with a graphical interface. For additional options, see the "Command Line Options" section.

## Operation Modes

SISTEMA_EMG has four main operation modes:

### 1. Simulation Mode

This mode allows you to experiment with the system using simulated EMG signals or public databases, without the need for hardware.

**How to use:**
1. Select "Simulation" in the interface
2. Choose the data source:
   - **Ninapro**: Database for sEMG and kinematics
   - **EMG-UKA**: EMG Database from University of Koblenz-Landau
   - **PhysioNet**: EMG Database for Gesture Recognition
   - **Synthetic**: Algorithmically generated signals
3. Adjust the noise level as needed
4. Click "Start" to begin the simulation
5. Observe the signal visualization and virtual prosthesis control

### 2. Collection Mode

This mode allows you to collect your own EMG data for personalized training.

**How to use:**
1. Select "Collection" in the interface
2. Choose the gesture to be collected:
   - Rest
   - Open hand
   - Closed hand
   - Pinch
   - Point
3. Set the collection duration (recommended: 5 seconds per gesture)
4. Click "Start Collection" and perform the requested gesture
5. Repeat for all desired gestures
6. Export the collected data to CSV by clicking "Export Data"

### 3. Training Mode

This mode allows you to train machine learning models with your collected data.

**How to use:**
1. Select "Training" in the interface
2. Click "Load Data" and select the CSV file with the collected data
3. Select the models to be trained:
   - SVM (Support Vector Machine)
   - MLP (Multi-Layer Perceptron)
   - CNN (Convolutional Neural Network)
   - LSTM (Long Short-Term Memory)
4. Configure the training parameters:
   - Train/test ratio (recommended: 80/20)
   - Cross-validation (recommended: 5 folds)
   - Number of epochs (for neural networks)
5. Click "Start Training"
6. View the results and performance metrics
7. Export the trained models by clicking "Export Models"

### 4. Execution Mode

This mode allows you to control the prosthesis in real-time using the trained models.

**How to use:**
1. Select "Execution" in the interface
2. Click "Load Model" and select the trained model
3. For physical mode:
   - Select the Arduino serial port
   - Click "Connect"
4. Adjust the confidence threshold (recommended: 0.7)
5. Click "Start Execution"
6. Perform gestures to control the prosthesis (physical or virtual)
7. Observe the signal visualization and real-time classification

## Electrode Positioning

Correct electrode positioning is crucial for EMG signal quality:

1. **Skin preparation:**
   - Clean the area with isopropyl alcohol
   - Remove hair if necessary
   - Let the skin dry completely

2. **Positioning for hand prosthesis control:**
   - **Flexor carpi radialis**: Position the electrodes on the proximal third of the forearm, on the anterior face
   - **Extensor carpi radialis**: Position the electrodes on the proximal third of the forearm, on the posterior face

3. **Electrode orientation:**
   - Position the electrodes parallel to the muscle fibers
   - Maintain a distance of 2 cm between electrodes
   - Position on the muscle belly, avoiding myotendinous junctions
   - The reference electrode should be on an electrically neutral area (e.g., bony prominence)

## Command Line Options

SISTEMA_EMG offers several command line options to customize its execution:

```
python main.py [options]
```

Available options:
- `--mode {physical,simulated}`: Operation mode (physical or simulated)
- `--port PORT`: Arduino serial port (only for physical mode)
- `--dataset PATH`: Path to the dataset to be used in simulated mode
- `--model PATH`: Path to the pre-trained model to be loaded
- `--no-gui`: Runs the system without graphical interface (console mode)

Example:
```
python main.py --mode physical --port COM3
```

## Troubleshooting

### Arduino Connection Problems

**Problem**: The system does not detect the Arduino.
**Solution**: 
- Check if the Arduino is properly connected
- Confirm that the USB driver is installed
- Try a different USB port
- Verify that the serial port is correct

### Low Quality EMG Signals

**Problem**: EMG signals are noisy or weak.
**Solution**:
- Check electrode positioning
- Clean the skin again
- Check MyoWare sensor connections
- Replace electrodes if they are dry
- Adjust filtering parameters in the software

### Inaccurate Classification

**Problem**: The system does not correctly classify gestures.
**Solution**:
- Collect more training data
- Try different machine learning models
- Adjust the confidence threshold
- Check gesture consistency during training and execution
- Try normalizing EMG signals

## Maintenance and Care

### Electrodes

- Replace electrodes regularly (recommended: every session)
- Store electrodes in a cool, dry place
- Do not reuse disposable electrodes

### MyoWare 2.0 Sensor

- Clean contacts with isopropyl alcohol when necessary
- Avoid bending or twisting the sensor
- Store in a dry, protected place

### Arduino

- Keep firmware updated
- Protect against electrostatic discharge
- Avoid disconnecting during operation

## Additional Resources

### Public EMG Databases

- **Ninapro**: [http://ninapro.hevs.ch/](http://ninapro.hevs.ch/)
- **EMG-UKA**: [https://www.uni-koblenz-landau.de/en/campus-koblenz/fb4/ist/rgdv/research/datasetstools/emg-dataset](https://www.uni-koblenz-landau.de/en/campus-koblenz/fb4/ist/rgdv/research/datasetstools/emg-dataset)
- **PhysioNet**: [https://physionet.org/content/emgdb/1.0.0/](https://physionet.org/content/emgdb/1.0.0/)

### Scientific Documentation

For detailed information about the scientific foundations of the system, see the following documents in the `docs/articles/` folder:

- `scientific_foundations.md`: Principles of electromyography and prosthesis control
- `machine_learning_algorithms.md`: Details about the implemented algorithms
- `signal_processing.md`: EMG signal processing techniques

## Support and Contact

For questions, suggestions, or collaborations, contact us through:
- Email: your-email@example.com
- GitHub: [your-username](https://github.com/your-username)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
