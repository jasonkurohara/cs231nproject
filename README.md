# cs231nproject
\documentclass[10pt,twocolumn,letterpaper]{article}

\usepackage{cvpr}
\usepackage{times}
\usepackage{epsfig}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}

% Include other packages here, before hyperref.

% If you comment hyperref and then uncomment it, you should delete
% egpaper.aux before re-running latex.  (Or just hit 'q' on the first latex
% run, let it finish, and you should be clear).
\usepackage[breaklinks=true,bookmarks=false]{hyperref}

\cvprfinalcopy % *** Uncomment this line for the final submission

\def\cvprPaperID{****} % *** Enter the CVPR Paper ID here
\def\httilde{\mbox{\tt\raisebox{-.5ex}{\symbol{126}}}}

% Pages are numbered in submission mode, and unnumbered in camera-ready
%\ifcvprfinal\pagestyle{empty}\fi
\setcounter{page}{4321}
\begin{document}

%%%%%%%%% TITLE
\title{Identifying Traffic Congestion Patterns Using Aerial Remote Sensing}

\author{Jason Kurohara\\
Stanford University\\
Stanford, CA\\
{\tt\small jkuro@stanford.edu}
% For a paper whose authors are all at the same institution,
% omit the following lines up until the closing ``}''.
% Additional authors and addresses can be added with ``\and'',
% just like the second author.
% To save space, use either the email address or home page, not both
\and
Allen Zhu\\
Stanford University\\
Stanford, CA\\
{\tt\small allenzhu@stanford.edu}
}

\maketitle
%\thispagestyle{empty}

%%%%%%%%% ABSTRACT
\begin{abstract}
   Rising traffic congestion is an inescapable problem in large urban areas of the world. In order to plan routes and reduce unnecessary travel time, governments require accurate and reliable information on vehicles in real time. Additionally, the rapid growth and commercialization of drone technology has led to an abundance of video and images of traffic scenes from an aerial view that fuel research and numerous applications in computer vision. This paper lies at the intersection of drone technology and monitoring traffic; we use a convolutional neural network to identify and track cars in sequential images trained on a variety of scenes captured by an Unmanned Aerial Vehicle (UAV). We compare our approach to baseline results produced by \textit{Fast Region Convolutional Network} (Fast-RCNN) architecture followed by Markov Decision Processes (MDP). Our approach uses the You-Only-Look-Once (YOLOv3) real-time object detection system and Simple Online and Real-time Tracking (SORT) algorithm to improve object detection and tracking of vehicles from an aerial view. We also compare the efficacy of Markov Decision Processing (MDP) as an object tracker to SORT. In this paper, we show that our network outperforms state-of-the-art baseline models in many metrics such as average precision, recall, F1, MOTA scores, and more.
\end{abstract}

%%%%%%%%% BODY TEXT
\section{Introduction}

Developments in drone technology and artificial intelligence intersect at the demand for intelligent unmanned aerial systems. In the global drone industry, UAVs have been applied in many areas such as security and surveillance, agriculture, and 3D mapping. In comparison to other types of camera platforms, UAV cameras come with their own separate set of challenges: high density as a result of wide angle views, small object detection due to the high cruising altitude of drones, and camera motion. Convolutional neural networks and recurrent neural networks are used in tandem to tackle the high level semantical tasks in computer vision such as object recognition and tracking of people, vehicles, buildings, and monuments.

Unmanned Aerial Vehicles (UAVs) are excellent mobile platforms for collecting high resolution images and videos. Satellite images do not provide the same quality of temporal data but still offer high density, small object images.

This paper describes our approach to accurately identify and track cars on roads in comparison to other state-of-the-art models. The goal is to be able to recognize the same car in multiple frames of a video. This will allow us to identify moving vehicles and their trajectory, which can aid measuring traffic.

We create a neural network to detect all visible vehicles (cars, trucks, buses, etc.) and create bounding boxes to segment an image. Additionally, we track the detected vehicles through videos in a variety of scenes (weather conditions, camera view, flying altitude, etc). 

We test two approaches to achieve this goal. Our first model uses the YOLOv3 architecture (a fast and reliable object detector) which then feeds a bounding box and class for each detected objects to the Simple Online Realtime Tracking (SORT) algorithm that matches detected objects across frames. Our second model uses the YOLOv3 architecture which then feeds a bounding box and class for each detected object to a MDP (Markov Decision Processing). 

The workflow is described as follows:
we start by feeding a 1024 by 540 pixel image into the YOLOv3 model \cite{yolov3}. The YOLOv3 network is configured to detect 80 different classes, but our model is only trained to detect three classes (truck, motorbike, car). We consider a vehicle as classified correctly if it passes an confidence threshold/intersection over union threshold of 0.5. 

After our trained YOLOv3 network outputs a prediction for the bounding box and class, we feed this output into Simple Online and Realtime Tracking (SORT) \cite{7533003}. This is a simple online tracking framework that focuses on frame-to-frame prediction and association to track the detected vehicles through time. SORT tracking performance is evaluated using Multi Object Tracking (MOT) benchmark.  \cite{journals/corr/Leal-TaixeMRRS15}.

Our second approach differs when the output of the YOLOv3 network is fed into a MDP. MDPs differ from SORT in that MDPs are trained using reinforcement learning that treats the appearance and reapperance of targets by treating them as state transitions in the MDP \cite{xiang_iccv15}. SORT and MDP output an object ID through the sequential data and its bounding box. Both these techniques are discussed in more detail later in the Methods section of this paper.

We evaluate the performance of our models to benchmark results produced by Du et al \cite{Du_2018_ECCV}.
%---------------------------------------------------------------------------------
\section{Related Work}

We divide the task into two parts: object detection and object tracking. There are many models for these tasks as described by the Unmanned Manned Aerial Vehicle Benchmark: Object Detection and Tracking by Du et al \cite{Du_2018_ECCV}, which include Faster-RCNN, R-FCN, RON, and SSD. 
As mentioned earlier in this paper, YOLOv3 is an object detection system targeted for real time processing. YOLOv3 has 53 convolutional layers for performing feature extraction and uses binary cross-entropy loss to output a bounding box and label of a detected object. In theory, YOLOv3's Average Precision (AP) metric is similar but 3 times faster in comparison to other state of the art object detectors. We decided to use this framework for its simple implementation, high AP in comparison the other models, and speed. For more information on its technical aspects, see the YOLOv3 paper here \cite{journals/corr/abs-1804-02767}. 

\begin{figure}[h]
    \centering
    \includegraphics[width=0.4\textwidth]{yolograph.png}
    \caption{YOLOv3 performance compared to other architectures.}
    \label{fig:uavdt}
\end{figure}

Unmanned Manned Aerial Vehicle Benchmark: Object Detection and Tracking \cite{Du_2018_ECCV} provides benchmark metrics on state-of-the-art models performing object detection and tracking on our dataset. The best reported model is an R-FCN as the object detector and SORT algorithm for the tracker. The R-FCN tracker received a 34.35\% mean average precision (mAP) score. The R-FCN and SORT model reported a 43.0 MOTA score and 61.5 IDF score. We compare our two models to these results and other useful metrics.




%---------------------------------------------------------------------------------
\section{Methods}
In this section, we discuss the YOLOv3 architecture, MDP, and SORT algorithms in more detail. 

\subsection{YOLO}
The first step of the task is to perform object detection in images. Multiple objects are identified in the image, classified under 80 class labels, and a location is determined by a bounding box on the image. Since the Unmanned Manned Aerial Vehicle Benchmark: Object Detection and Tracking \cite{Du_2018_ECCV} paper did not attempt to use the YOLO architecture, we wanted to see if this state-of-the-art, fast, real-time object dectection algorithm would outperform the architectures proposed in the paper.

So, we trained the YOLOv3 architecture on the UAVDT data set. This dataset is described in greater detail in section 4 of this paper. A vehicle is classified correctly if its intersection over union threshold (the ratio of area of overlap of the predicted and ground truth labels to total area of the predicted and ground truth boxes) is greater than 0.5. 

\begin{figure}[h]
    \centering
    \includegraphics[width=0.4\textwidth]{qwerty.png}
    \caption{YOLOv3 Framework}
    \label{fig:uavdt}
\end{figure}

Other object detection systems like R-FCN and R-CNN apply the model to an image at multiple locations and scales, and high scoring regions of the image are considered detections. However,  YOLO divides the image into $SxS$ regions and predicts the bounding boxes and probabilities for each region, where the $B$ bounding boxes are weighted by the predicted probabilities. YOLO is one of the fastest algorithms for object detection because it looks at the entire image at test time and its predictions are informed by global context in the image. Additionally, YOLO makes bounding box predictions with a single network evaluation unlike R-CNN and Faster R-CNN which require thousands for a single image. 

As mentioned, YOLO divides the input image into an $S$x$S$ grid, and if the center of the object falls into a grid cell, that grid cell is responsible for detecting that object. Each grid predicts $B$ bounding boxes and confidence scores for those boxes. The confidence scores reflect how confident the model is that the box contains and object and how accurately it predicted it. 

\begin{figure}[h]
    \centering
    $\text{ confidence score = }Pr(Object) \cdot IOU_{pred}^{truth}.$
    \newline
    \caption{The YOLO model divides the image into an $SxS$ grid and for each grid cell predicts $B$ bounding boxes and a confidence score \cite{DBLP:journals/corr/RedmonDGF15}.}
    \label{fig:yolo}
\end{figure} 

 Each bounding box prediction contains a $x,y,w,h$ and confidence score. $(x,y)$ is the center of the box and $(w,h)$ are the width and height of the predicted box. Each grid cell also predicts $C$ conditional class probabilities as $Pr(Class_i|Object)$, which are probabilities conditioned on the grid cell containing an object.

\begin{figure}[h]
    \centering
    \includegraphics[scale=0.22]{yolodiagram.png}
    \caption{The YOLO model divides the image into an $SxS$ grid and for each grid cell predicts $B$ bounding boxes and a confidence score \cite{DBLP:journals/corr/RedmonDGF15}.}
    \label{fig:yolo}
\end{figure}

The final layer of the YOLOv3 network predicts both class probabilities and bounding box coordinates. Bounding box coordinates are normalized by the image height and width, along with the bounding box width and height so that values are bounded between 0 and 1. Additionally, the last connected layer uses a leaky rectified linear activation function described in Stanford's CS231n lecture slides

\begin{figure}[h]
    \centering
    $f(x) = \vec{1}(x < 0)(\alpha x) + \vec{1}(x >= 0)(x) \text{ where } \alpha = 0.1$
   % \caption{Leaky ReLU function}
    \label{fig:yolo}
\end{figure}

YOLO optimizes the model by using the following loss function. The parameters $\lambda_{coord}=5$ and $\lambda_{noobj}=0.5$ are regularization terms that decrease the loss form confidence predictions that do not contain objects and increase the loss from bounding box coordinate predictions.

\begin{figure}[h]
    \centering
    \includegraphics[scale=0.35]{yololoss.png}
    \caption{$\vec{1}_i^{obj}$ denotes if the object appears in cell $i$ and $\vec{1}_{ij}^{obj}$ denotes that the jth boudning box predictor in cell $i$ is "responsible" for that prediction. \cite{DBLP:journals/corr/RedmonDGF15}.}
    \label{fig:yolo}
\end{figure}

At test time, we multiply the class probabilities and individual box confidence predictions, 

\begin{figure}[h]
    \centering
    $Pr(Class_i|Object) \cdot Pr(Object) \cdot IOU_{pred}^{truth} = Pr(Class_i) \cdot IOU_{pred}^{truth}$
    \caption{These scores encode the probability of the class appearing in the grid cell and how well the box compares to the ground truth.}
    \label{fig:yolo}
\end{figure}

For more information, see the original YOLO paper here \cite{journals/corr/abs-1804-02767}.

\subsection{Markov Decision Processes (MDPs)}
Markov Decision Processes or MDPs use reinforcement learning to track objects through time. The lifetime of a target is modeled as a MDP that consists of a target state $s \in S$ which is the status of the target, an action $a \in A$ that is performed on the object, a state transition function $T:S$ x $A \rightarrow S$ that describes the effect of each action on each state, and a reward function $R: S$ x $A \rightarrow R$ after executing action $a$ to $s$.

Each object is in state $S_{active}, S_{tracked}, S_{Lost}, S_{inactive}$ which are subspaces of $S$. The seven transition functions model the actions that can be performed on any given state. For instance, $a_4$ on a tracked target transitions the object into $S_{Lost}$.

\begin{figure}[h]
    \centering
    \includegraphics[scale=0.5]{markovdiag.png}
    \caption{Visual representation of a MDP for one object \cite{xiang_iccv15}.}
    \label{fig:yolo}
\end{figure}

While the technical details of the Markov Decision Process are out of scope of this paper, after learning the policy/reward function of the MDP using reinforcement learning, we dedicate a MDP for each object. Thus, given sequential video frames, targets are tracked in states. Additionally, we initialize a MDP for each object detection which is not covered by any tracked targets to account for multiple objects. For more information, see Yu Xiang's paper on Multi-Object Tracking by Decision Making \cite{xiang_iccv15}.

\subsection{Single, Online, Real-time Tracking (SORT)}

SORT is a multi-object tracker suggested by Alex Bewley that utilizes a combination of a Kalman Filter and Hungarian algorithm \cite{Kuhn55thehungarian} implementation as motion prediction and data association components. In SORT, appearance features beyond the detection component are ignored, and only the bounding box position and size are used for both motion estimation and data association. The state of each target is modeled as:

\begin{figure}[h]
    \centering
    $\vec{x}=[u,v,s,r,\dot{u},\dot{v},\dot{s},]$
   % \caption{SORT state}
    \label{fig:sort_state}
\end{figure}

Here, $u$ and $v$ represent the horizontal and vertical pixel location of the target center, $\dot{u}$,$\dot{v}$ represent the target's horizontal and vertical velocities, and $s$ and $r$ represent the area scale and the aspect ratio of the target’s bounding box. Upon association of a detection with a target, the detected bounding box is used to update the target state in which the velocities are solved with a Kalman filter \cite{Welch:1995:IKF:897831}. If there is no association the linear velocity model is used to update the target state. To assign detections to targets, an assignment cost matrix is  computed as the IOU distance between each detection and all predicted bounding boxes from the Kalman filter. The Hungarian algorithm is used to optimally solve this assignment. SORT has a MOTA score of 33.4\% when trained on the MOT benchmark sequences \cite{journals/corr/Leal-TaixeMRRS15}.

\subsection{Metrics}

First to evaluate our detection model, we calculate the precision, recall, F1 score and average precision on the test set. Precision, recall, and F1 score are the usual derived measures, $P = T P/(T P + F P)$, $R = T P/(T P + F N)$, and $F1=2PR/(P+R)$. Average precision is the Area Under the Curve (AUC) average precision and is computed by

\begin{equation}
AP = \sum((r_{n+1}-r_n)p_{itp}(r_{n+1}))
\end{equation}
\begin{equation}
p_{itp}(r_{n+1})=\max_{\Tilde{r}\geq r_{n+1}}p(\Tilde{r})
\end{equation}

where $r_n$ and $r_{n+1}$ are adjacent sampled recall value points, and $p_{\Tilde{r}}$ is the sampled precision at recall $\Tilde{r}$ \cite{pascal-voc-2012}. 

In order to evaluate our model on multi-object tracking, we utilize the standard MOT Benchmark evaluation metrics \cite{journals/corr/Leal-TaixeMRRS15}\cite{DBLP:journals/corr/RistaniSZCT16}. These include identification precision (IDP), identification recall (IDR), and the corresponding F1 score IDF1 (the ratio of correctly identified detections over the average number of ground-truth and computed detections), Multiple Object Tracking Accuracy (MOTA), and Multiple Object Tracking Precision (MOTP). For each tracked target, we also classify them as Mostly Tracked targets (MT, the number of ground truth trajectories that are covered by a track hypothesis for at least 80 percent of their lifespan), Partially Tracked targets (PT, the number of ground truth trajectories that are covered by a track hypothesis for at least 20 percent of their lifespan), and Mostly Lost targets (ML, the number of ground truth trajectories that are covered by a track hypothesis for less than 20 percent of their lifespan). For each sequence we track we also compute the total number of False Positives (FP), the total number of False Negatives (FN), the total number of ID Switches (IDS) between frames, and the total number of times a trajectory is Fragmented (FM).

\begin{equation}
    \centering
    MOTA = 1-\frac{\sum_{t}(FN_t+FP_t+IDSW_t)}{\sum_{t}GT_t}
    \label{fig:MOTA}
\end{equation}
\begin{equation}
    \centering
    MOTP = \frac{\sum_{t,i}d_{t,i}}{\sum_{t}c_t}
    \label{fig:MOTP}
\end{equation}
\begin{equation}
    \centering
    IDP = \frac{\sum_{t}TP_t}{\sum_{t}(TP_t+FP_t)}
    \label{fig:IDP}
\end{equation}
\begin{equation}
    \centering
    IDR = \frac{\sum_{t}TP_t}{\sum_{t}(TP_t+FN_t)}
    \label{fig:IDR}
\end{equation}
\begin{equation}
    \centering
    IDF1 = \frac{2\sum_{t}TP_t}{\sum_{t}(2TP_t+FP_t+FN_t)}
    \label{fig:IDR}
\end{equation}

In addition to the aforementioned variable explanations, $TP$ refers to true positives, $t$ is the frame index, $GT$ is the number of ground truth objects, $c_t$ denotes the number of target matches in frame $t$ and $d_{t,i}$ is the bounding box overlap of target $i$ with its ground truth object. 


%---------------------------------------------------------------------------------

\section{Dataset and Features}

\subsection{Training and Validation Dataset}
To train and validate our model, we utilize the UAVDT benchmark provided by Du et al \cite{Du_2018_ECCV}. The UAVDT benchmark consists of 100 video sequences, which are selected from over 10 hours of videos taken with a UAV platform at a number of locations in urban areas, representing various common scenes including squares, arterial streets, toll stations, highways, crossings and T-junctions. The videos are recorded at 30 frames per seconds (fps), with the JPEG image resolution of 1080 x 540 pixels. About 80, 000 frames in the
UAVDT benchmark data set are annotated  with over 2,700 vehicles with 0.84 million
bounding boxes. Regions that cover too small vehicles are ignored in each frame due to low resolution. We split the data into training, validation, and testing with an 80-10-10 split. 

Each video sequence is annotated with information about each vehicle in each frame of the video. For a certain car of interest in a frame these annotations document the frame number, car id number, and bounding box information for each car. The car id is used to track cars from one frame to subsequent frames. The same car will have the same car id throughout the video sequence.

After processing and formatting the data, a sample processed picture with a bounding box from the ground truth annotation labels and an example of an annotation label for which object detection are shown below. 

\begin{figure}[h]
    \centering
    \includegraphics[scale =0.20]{img000006.jpg}
    \caption{Our dataset is labels a ground truth for the bounding box coordinates and size. In this image, we visualize the bounding box around each vehicle.}
    \label{fig:uavdt}
\end{figure}

In comparison to other datasets, the UAVDT dataset provides higher object density with moving vehicles (as opposed to parked vehicles) and is captured from various weather conditions, flying attitudes, and camera views.
%---------------------------------------------------------------------------------
\section{Experiments/Results/Discussion}
We evaluate our model based on its ability to detect objects and also track them.

\subsection{Benchmark}
The benchmark model we use comes from the UAVDT paper \cite{xiang_iccv15}. The model that received the best MOTA score of 43.0\% is an FrCNN as the object detector and MDP as the object tracker. Although this model received the highest MOTA score, the FrCNN only had an average precision of 22.32\%. The FrCNN performed worse than the R-FCN object tracker that had an average precision of 34.35\%. Despite the R-FCN's higher average precision, its combination with the various object trackers tested in the UAVDT paper yield a lower MOTA score than the FrCNN and MDP model. Therefore, our benchmark model is the FrCNN and MDP.

\subsection{Object Detection Results}
For training, we began with untrained weights of the YOLOv3 network configured detect the 80 classes from the COCO dataset \cite{10.1007/978-3-319-10602-1_48}. Figure 9 shows the loss over 63,000 images. 

\begin{figure}[h]
    \centering
    \includegraphics[scale =0.4]{loss.png}
    \caption{We graph the loss over around 63,000 images (8 batches). The training set contains 40 scenes that vary in illumination, altitude, angle, etc.}
    \label{fig:uavdt}
\end{figure}

Before feeding the images into YOLOv3, the images are normalized and resized to a 416 x 416 image. We use a batch size of 8 images, momentum value of 0.9, and a decay rate of 0.0005.

Our model is configured to detect 80 different classes. However, changing the number of classes to only detect cars, trucks and motorbikes (the only classes in our dataset) did not significantly improve the evaluation metrics. So, we trained the 80 class object detector on the dataset with only the three types of objects.

\begin{figure}[h]
    \centering
    \includegraphics[scale =0.20]{yolooutput.png}
    \caption{Our dataset only labels cars, trucks, and motorbikes. After training for 7 epochs, we see that our model detected most of the vehicles in the image along with a truck and motorbike.}
    \label{fig:uavdt}
\end{figure}

The evaluation metrics on the validation set after training for 7 epochs are listed in Figure 11.

\begin{figure}[h]
    \centering
    \includegraphics[scale =0.24]{yoloresults.png}
    \caption{After each epoch, we test on a validation set to generate the evaluation metrics. We used the best weights from epoch 6 to test our model.}
    \label{fig:uavdt}
\end{figure}

After training our model to find the best weights, we evaluate the model on 4 different test scenes called M103, M104, M105, and M106. Each scene is recorded over 40 seconds.

\begin{figure}[h]
    \centering
    \includegraphics[scale =0.23]{testyolo.png}
    \caption{The T-intersection and Highway are the type of scenes that are captured by the drone. The runtime is the number of frames evaluated over one second.}
    \label{fig:uavdt}
\end{figure}

The YOLOv3 model performs detections better than all of the object detectors suggested in the UAVDT paper, with higher Average Precision scores for all of the tested scenes by at least 10\% and at most 26.45\% compared to the UAVDT paper's R-FCN, the best detector mentioned in the paper. Other state-of-the-art detectors include Single Shot MultiDetector (SSD), Faster R-CNN, and Reverse Object Connection with Prior Networks (RON) which had AP scores of 33.62\%, 22.32\%, and 21.59\%, respectively. YOLOv3 outperforms all these models.

 We also evaluate our model's run-time performance. Using 1 vCPU with 6 GB memory and a NVIDIA Tesla K80 GPU, the speed of the you-only-look-once algorithm truly shines. R-FCN and other architectures from the UAVDT benchmark ran at less than 5 fps, which is far too slow for practical applications. Our model, on the other hand, achieves an average run-time of 16.05 fps, which is 2x faster than the benchmark R-FCN and usable on a real-time video feed. This is a significant improvement from the benchmark model.

To see evaluation metrics from the UAVDT paper in more detail, see \cite{xiang_iccv15}.

\subsection{Object Tracking Results}
The next part of this paper discusses the evaluations on object tracking. After the YOLOv3 network outputs the bounding box coordinates and confidence score, the results are fed into a multi-object tracker. We utilized both the SORT and MDP trackers. We compare our results using the SORT and MDP trackers to the benchmark FrCNN and SORT model and the benchmark FrCNN and MDP model.

In order to train the MDP tracker, we pick 4 random training scenes from our training set. We used few training scenes due to constrained resources for this project. Despite using limited training data, our results using the MDP tracker are significantly better than our results from the SORT tracker. 

\begin{figure}[h]
    \centering
    \includegraphics[scale =0.45]{SORT.png}
    \caption{MOT Metrics computed from feeding the YOLOv3 detections into the SORT tracker.}
    \label{fig:sort}
\end{figure}
\begin{figure}[h]
    \centering
    \includegraphics[scale =0.45]{MDP.png}
    \caption{MOT Metrics computed from feeding the YOLOv3 detections into the MDP tracker.}
    \label{fig:sort}
\end{figure}

From these results it is evident that compared to the benchmark model, in addition to generating better detections, YOLOv3 in combination with both the SORT and MDP trackers is also better at multi-object tracking. The best detector with SORT that was tested in the UAVDT paper was FrCNN with SORT, which achieved a 39.0\% MOTA score and a 43.7\% IDF1 score. Averaged over our four test scenes, YOLOv3 with SORT achieves a 40.6\% MOTA score and a 44.3\% IDF1 score. Compared to FrCNN with SORT, YOLOv3 with SORT is more accurate with detections running 3 times faster. YOLOv3 with SORT outperforms all of the combinations of detectors with SORT tested in the paper. In using the MDP tracker, the benchmark set by the UAVDT paper, FrCNN with MDP, achieves a MOTA score of 43.0\% and 61.5\% IDF1 score. Averaged over our four test scenes, YOLOv3 with MDP achieves a 48.0\% MOTA score and a 64.2\% IDF1 score. The YOLOv3 with MDP combinations outperforms all of the detector-tracker combinations in the UAVDT paper. 

\begin{figure}[h]
    \centering
    \includegraphics[scale =0.45]{img_000139.png}
    \label{fig:top}
\end{figure}
\begin{figure}[h]
    \centering
    \includegraphics[scale =0.45]{img_000159.png}
    \caption{Comparison of the detection associations computed by SORT from the bounding boxes input to it by YOLOv3. The top shows frame 139 from the M1201 sequence, while the bottom shows frame 159 also from the M1201 sequence. Boxes of the same color are associations of the same target through the frames. }
    \label{fig:bottom}
\end{figure}

%---------------------------------------------------------------------------------
\section{Conclusion/Future Work}
This paper presents a fast, accurate, and reliable model to detect and track cars from an aerial view. The applications to this technology are ubiquitous because there are many use cases in the areas of surveillance, military, commercial, and urban planning. 

The YOLOv3 network outperforms other state-of-the-art models like R-FCN, Faster R-CNN, and SSD in both run-time and accuracy/precision. YOLOv3 runs 3x faster than other models at 16.05 fps, making it  useful for real-time applications on aerial platforms and other embedded systems. For object tracking, having the fastest  detector is critical in order to pass detections to the tracker. With object tracking, the MDP tracker outperforms SORT, with a higher MOTA score of over ten percent in some scenes. However, in order to perform real-time multi-object tracking, the MDP tracker is too slow to be of use. On the M1303 test sequence for example using 1 CPU, SORT processed 109.5 frames per second while MDP only processed 0.51 frames per second. Further, MDP must first be trained, while SORT does not. We recommend using YOLOv3 and SORT to perform real-time object tracking for its combination of accuracy and speed. 

In the future, we would like our model to run at least 30 fps in order to be used on UAVs. Another useful improvement is to use a smaller network that achieves similar scores as YOLOv3 and MDP, since these models are quite large and bulky on mobile platforms. Additionally, we would like to perform object detection and tracking on objects other than trucks, motorbikes, and cars to increase the practicality of our model.
%---------------------------------------------------------------------------------

%---------------------------------------------------------------------------------
\section{Contributions and Acknowledgments}
J.K. adapted the YOLOv3 network, trained the model, tuned hyper-parameters, and processed image and label datasets. A.Z. adapted the SORT and MDP trackers to perform multi-object tracking, trained the MDP tracker, and evaluated MOT metrics. J.K. and A.Z. wrote the paper.
This project has not been submitted to a peer-reviewed conference or journal.

\subsection{Starter Code}
The YOLOv3 algorithm was implemented by Erik Linder-Norén \cite{yolov3}. We made minor changes in order to fit the model into our pipeline and datasets. We did start by using pretrained weights and trained the model on the UAVDT dataset \cite{xiang_iccv15}. We also did not write the source code for the SORT algorithm. The implementation can be found on Alex Bewley's github \cite{7533003}. MOT metrics are implemented using the python MOT metrics library. The MDP source code was implemented by Yu Xiang. Yu Xiang's implementation can be found here \cite{xiang_iccv15}. The YOLOv3 model is implemented in PyTorch \cite{paszke2017automatic}. 



%-------------------------------------------------------



{\small
\bibliographystyle{ieee}
\bibliography{egbib}
}

\end{document}
