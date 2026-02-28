# ANN-Internal-Analysis
1️⃣ Clone the Repository
git clone https://github.com/your-username/ANN-Internal-Analysis.git
cd ANN-Internal-Analysis
2️⃣ Create a Virtual Environment (Recommended)

Using venv:

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
3️⃣ Install Dependencies

If a requirements.txt file is included:

pip install -r requirements.txt

If not, install the core libraries manually:

pip install torch torchvision matplotlib numpy jupyter
4️⃣ Launch Jupyter Notebook
jupyter notebook

Open the main notebook file (e.g., ANN_Internal_Analysis.ipynb).

5️⃣ Run the Notebook Cells in Order

Make sure to:

Set the device (cpu or cuda)

Load the dataset (e.g., MNIST)

Initialize the model

Train the model

Run the activation hook cells

Execute internal weight analysis cells

Run all cells sequentially from top to bottom to ensure proper execution.

6️⃣ (Optional) Run as Python Script

If a .py training script is provided:

python train.py
7️⃣ Expected Outputs

After running successfully, you should see:

Training loss printed per epoch

Model accuracy on test data

Activation statistics from hooks

Weight update statistics (e.g., delta mean/std/max)

Visualizations of internal representations

🖥 Hardware Support

Works on CPU

Automatically uses GPU (CUDA) if available

Check CUDA availability:

import torch
print(torch.cuda.is_available())
