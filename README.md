# QML
Quantum Computing

   Started exploring in QML from 1 Feb 2025, And today 4 feb I am creating repo for my QML coding/Projects.

Directory structure:
mkdir src data results notebooks results/model_checkpoints results/logs
touch src/__init__.py src/quantum_model.py src/train.py src/test.py README.md run.sh


quantum_ai_project/  
│── data/                     
│   ├── train_data.csv        # Training dataset ✅  
│   ├── test_data.csv         # Testing dataset ✅  
│   ├── validation_data.csv   # Validation dataset ✅ (Newly added)  
│── src/                      
│   ├── quantum_model.py      # Quantum Neural Network ✅  
│   ├── train.py              # Training script (Updated to split data) ✅  
│   ├── test.py               # Model evaluation ✅  
│   ├── utils.py              # Helper functions (data split, logging) ✅ (Newly added)  
│── results/                   
│   ├── model_checkpoints/     # Model storage  
│   │   ├── qnn_model_v1.pth   # First trained model ✅  
│   │   ├── qnn_model_v2.pth   # New model version ✅  
│   ├── logs/                  # Training & evaluation logs  
│   │   ├── metrics.txt        # Model performance ✅  
│   ├── hyperparams.json       # Model hyperparameters ✅  
│   ├── visualizations/        # Graphs & plots  
│   │   ├── loss_curve.png     # Training loss visualization ✅  
│── notebooks/                # Jupyter Notebooks for exploration ✅  
│── README.md                  # Project description ✅  
│── requirements.txt           # Dependencies ✅  
│── run.sh                     # Shell script to run the project ✅  
│── venv/                      # Virtual environment ✅  
