# Real chessboard to FEN
Application that leverages computer vision to detect chess pieces disposition and write it into FEN notation.

I made this work for my IoT University project.
I took inspiration from the work of @shainisan. Repo **[here](https://github.com/shainisan/real-life-chess-vision?ref=blog.roboflow.com)**.

## Usage
Download through the command:
```bash
git clone https://github.com/aledevv/real-chessboard-to-fen.git
```

- The notebook _app.ipynb_ contains all the code for detection. 

- _images_ folder contains pictures to process.

> [!WARNING]
> **Weights are tuned using my chessboard and my pieces**.
> If you want to use yours you have to fine-tune a new model (this applies to both corner and pieces detection)

> [!TIP]  
> I found useful **[this notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb)** to train the YOLOv8 model.

## Detection process
1. User takes a photo of their chessboard
   
    ![User image](assets/photo.jpeg)
3. A first model detects corners of the chessboard
   
    ![Corners](assets/corners.png)
4. Image is cropped and transformed from 3D to 2D and squares position is computed (red grid)
    ![Cropped and Transformed](assets/transformed.jpg)
    > Grid is not perfectly aligned to the squares because the camera position for the detection was not been decided yet, so I created an __offset__ variable to adjust the grid afterwards.
5. Pieces are detected

    ![Detection](assets/detection.png)
6. Prediction boxes are intersected with the grid to find each piece position on the board.
7. FEN is written and can be exported to Lichess or a Stockfish API.
   
    ![Lichess FEN chessboard](assets/lichess_fen.png)

The same process is described in the repo linked above.

## Training
I used a YOLOv8n for corners detection and a YOLOv8s for pieces recognition.
### Corners
The training leaded to a mAP of 80.5%, it could be improved but it was enough for my needs.

**Confusion matrix**
![](assets/training/corners/conf_matrix.png)
**PR curve**
![](assets/training/corners/pr_curve.png)

### Piece detection
The model reached a mAP of 96.7%.

**Confusion matrix**
![](assets/training/pieces/confusion_matrix.png)
**Plots**
![](assets/training/pieces/plots.png)
**Batch training**
![](assets/training/pieces/batch_test.jpeg)
