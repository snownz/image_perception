# Project Object Detection

## Objective:

- Identify and Classify Objects on the street scenario
- Create a near-perfect Labeled dataset
- Research and develop a Novel model capable of understanding not only RGB but also Depth
- Deliver a full Working Algorithm with the following outputs:
  - Detections + Classes + Position
  - Image Math Features for another model's integration
  - Non-detected Objects based on the Depth

## Propouse
- Detected Objects can be used for planning in autonomous driving
- Detected objects may help with robot tasks ( snow removal, sidewalk inspection, etc. )
- Obstacle Avoidance 
- Path planning
- Richer feature description for all other AI models

## Non-Detected Objects
- Use the depth image to avoid non-detected objects

## Next Steps
* Setup a Server with 2 GPUS
* Train the proposed models
* Evaluate:
  - Training time
  - Training Precision and Accuracy
  - Inference Time
  - Model Integration
* Integrate the better model into an Algorithm
* Finish the package Model + Algorithm

<br><br><br><br><br><br><br><br><br><br>

# **Dataset Analysis Report**

## **Dataset Overview**
- **Total Number of Samples:** 1700 
- **Classes:** Person, Birds, Parking Meter, Stop Sign, Street Sign, Fire Hydrant, Traffic Light, Motorcycle, Bicycle, LMVs, HMVs, Animals, Poles, Barricades, Traffic Cones, Mailboxes, Stones, Small Walls, Bins, Furniture, Pot Plant, Sign Boards, Boxes, Trees.  

---

## **1. Class Distribution**
### **Observations:**
- The dataset exhibits class imbalance.  
- **LMVs (Light Motor Vehicles)** and **Animals** are the most represented classes.  
- Underrepresented classes include **Stop Signs, Parking Meters, and Boxes**.  

### **Impact on Generalization:**
- The model might overfit dominant classes while struggling to generalize for rare ones.  

![Class Distribution](images/im1.png)  
*Figure: Total Count of Each Class*  

---

<br><br><br><br><br><br><br><br>

## **2. Spatial Distribution of Objects**
### **Observations:**
- **Poles, Barricades, and Traffic Cones** cluster around certain spatial areas.  
- **Mailboxes and Motorcycles** are sparsely located.  

### **Impact on Model Performance:**
- Biases in spatial distribution may lead to model overfitting to frequent locations.  
- Rare class spatial presence may affect detection accuracy.  

![Heatmaps1](images/im2.png) 

![Heatmaps2](images/im3.png) 
*Figure: Class-wise Heatmaps of Object Positions*  

---

<br><br><br><br><br><br><br><br>
<br><br><br><br><br><br><br><br>
<br><br><br><br>

## **3. Class Diversity Across Images**
### **Observations:**
- A large proportion of images contain only one or two unique object classes.  
- Only a small percentage of images contain more than five unique object types.  

### **Impact on Generalization:**
- Limited diversity per image might restrict inter-class relationship learning.  
- May impact performance in real-world scenarios with multiple objects.  

![Class Diversity](images/im4.png)   
*Figure: Histogram of Unique Classes per Image*  

---

<br><br><br><br><br><br><br><br>
<br><br><br><br><br><br><br><br>


## **4. Number of Detections per Image**
### **Observations:**
- Most images contain fewer than five detections.  
- There are some outliers with over 50 detections, indicating annotation inconsistencies.  

### **Impact on Generalization:**
- Sparse annotations in most images could lead to weak model learning.  
- Outliers may cause models to overestimate object presence.  

![Detections per Image](images/im5.png) 
*Figure: Histogram of Detections per Image*  

---

<br><br><br><br><br><br><br><br>
<br><br><br><br><br><br><br><br>


## **5. Class Metrics: Average Count & Probability of Appearance**
### **Observations:**
- **LMVs and Animals** have the highest probability of appearance.  
- **Boxes and Stop Signs** have the lowest probability.  

### **Impact on Generalization:**
- The disparity in class probabilities makes achieving balanced performance challenging.  
- Overrepresented classes could dominate predictions.  

![Class Metrics](images/im6.png)  
*Figure: Average Count and Probability of Appearance*  

---

<br><br><br><br><br><br><br><br>
<br><br><br><br>


## **Conclusions**

### **Strengths**
1. **Well-represented dominant classes:** The dataset provides ample data for major classes.  
2. **Detailed class metrics:** The dataset contains statistics for spatial and count distributions.  

### **Weaknesses**
1. **Class Imbalance:** Some classes are heavily overrepresented while others are scarce.  
2. **Sparse Annotations:** Many images have very few object instances, limiting learning diversity.  
3. **Spatial Bias:** Some object types appear only in specific locations, affecting generalization.  

### **Recommendations**
- **Data Augmentation:** Use synthetic balancing techniques for rare classes.  
- **Better Sampling:** Ensure diverse class representation in training batches.  
- **Spatial Augmentation:** Randomize placements during training to reduce spatial bias.  

---

<br><br><br><br><br><br><br><br>
<br><br><br><br><br><br><br><br>
<br><br><br><br><br><br><br><br>


# **Model Research Report**

## **Introduction**
Object detection plays a critical role in autonomous driving, requiring high accuracy, real-time inference, and robustness to class imbalance and occlusions. This research evaluates multiple state-of-the-art models to identify the best-performing architectures for self-driving perception. The models analyzed include:

- **YOLOv11**
- **RT-DETR**
- **Faster R-CNN**
- **EfficientDET**
- **VGG**
- **AlexNet**
- **MASK R-CNN**
- **RetinaNet**
- **UNET**
- **ViT (Vision Transformer)**

## **Model Comparisons and Findings**

### **1. Accuracy & Performance**
- **RT-DETR** and **YOLOv11** achieve state-of-the-art detection accuracy (~54.7% mAP on COCO), making them suitable for real-time autonomous applications.
- **Faster R-CNN** maintains high precision (~40–45% mAP), but its recall is lower due to its two-stage architecture.
- **EfficientDET** balances speed and accuracy but underperforms compared to YOLOv11 and RT-DETR.
- **RetinaNet** applies focal loss to address class imbalance but lacks real-time processing capabilities.
- **VGG, AlexNet, and UNET** are outdated for modern object detection tasks.
- **ViT** shows promise with transformer-based global reasoning but requires significantly more computational resources.

### **2. Speed & Real-Time Suitability**
- **YOLOv11 and RT-DETR exceed 30 FPS**, making them ideal for real-time deployment in self-driving cars.
- **Faster R-CNN** is significantly slower (~5–10 FPS) due to its two-stage pipeline, making it less viable for real-time perception.
- **EfficientDET and RetinaNet** provide moderate frame rates but are not the best fit for self-driving applications.

### **3. Handling of Class Imbalance**
- **RT-DETR’s set-based query prediction** improves rare-class detection and reduces bias towards frequent objects.
- **YOLOv11’s focal loss and augmentation techniques** help balance object detection performance across rare and dominant classes.
- **Faster R-CNN uses proposal-based training**, which ensures detection of rare objects but requires additional loss weighting or resampling.

### **4. Occlusion Robustness**
- **RT-DETR** leads in occlusion robustness due to its transformer-based **global attention mechanism**, effectively detecting objects even when partially visible.
- **YOLOv11** performs well under occlusions using **feature pyramids and contextual enhancements**, but relies more on clear visual cues.
- **Faster R-CNN** handles occlusions by leveraging region proposals for partially visible objects, improving detection but at the cost of speed.

### **5. Multi-Scale Detection**
- **RT-DETR and YOLOv11** excel with feature fusion techniques that allow effective detection across a range of object sizes.
- **Faster R-CNN with Feature Pyramid Networks (FPN)** provides strong multi-scale detection capabilities but is computationally expensive.

### **6. False Positive Control**
- **Faster R-CNN has the lowest false positive rate**, as its two-stage pipeline ensures rigorous verification of detected objects.
- **RT-DETR eliminates NMS-related false detections** through its end-to-end query-based approach.
- **YOLOv11 minimizes false alarms** through refined objectness scoring and improved Non-Maximum Suppression (NMS) techniques.

---

## **Final Model Selection for Self-Driving Car Perception**
Based on the comprehensive analysis, the **top three models recommended for deployment** in autonomous driving are:

### **1. RT-DETR** (Best Overall Performance)
✅ **Highest accuracy** in detecting occluded and small objects.

✅ **Transformer-based detection** ensures strong spatial awareness.

✅ **End-to-end detection** eliminates duplicate predictions.

❌ Requires more computational resources compared to YOLOv11.

### **2. YOLOv11** (Best for Real-Time Processing)
✅ **Fastest inference speed** (~100+ FPS for smaller variants).

✅ **High accuracy with strong generalization** across object classes.

✅ **Optimized for edge deployment** and self-driving hardware.

❌ Slightly lower accuracy on heavily occluded objects than RT-DETR.

### **3. Mask R-CNN** (Best for Precision-Focused Applications)
✅ **High precision and minimal false positives.**

✅ **Effective for offline validation and object re-identification.**

❌ **Slow inference (~5–10 FPS), unsuitable for real-time self-driving.**

---

## **Deployment Recommendations**
- **Primary Model: RT-DETR** should be the first choice for detecting rare and occluded objects with high accuracy.
- **Real-Time Operations: YOLOv11** should be used for real-time perception due to its speed and efficiency.
- **Validation & Error Checking: Faster R-CNN** can be employed in offline evaluations or high-precision safety-critical tasks.

---

## **Conclusion**
For self-driving perception, **RT-DETR and YOLOv11 are the optimal choices**. **RT-DETR excels in detection accuracy and occlusion robustness**, while **YOLOv11 provides unmatched speed for real-time operations**. **Mask R-CNN remains useful for high-precision verification but is not viable for live vehicle deployment**. These models provide a strong balance between detection reliability, inference speed, and robustness in complex driving scenarios.

## **References**
1. **YOLOv11**: Ultralytics documentation and performance benchmarks.
2. **RT-DETR**: Baidu research on Real-Time Detection Transformer.
3. **Faster R-CNN**: Ren, S., He, K., Girshick, R., & Sun, J. (2015). "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks."
4. **EfficientDET**: Tan, M., Pang, R., & Le, Q. V. (2020). "EfficientDet: Scalable and Efficient Object Detection."
5. **VGG**: Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition."
6. **AlexNet**: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks."
7. **MASK R-CNN**: He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). "Mask R-CNN."
8. **RetinaNet**: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). "Focal Loss for Dense Object Detection."
9. **UNET**: Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation."
10. **ViT (Vision Transformer)**: Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."
11. **Self-Driving Perception Research**: KITTI, BDD100K, and nuScenes benchmarks.
12. **Transformer vs. CNN Object Detectors**: Research from arXiv on transformer-based detection.
13. **COCO Benchmarks**: Performance evaluation results from MS COCO leaderboards.