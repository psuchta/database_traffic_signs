close all; clear; clc;

load("detector.mat") % comes from yolov2.m

inputSize = [224, 224, 3];

I = imread('real_test\1.jfif');
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
 
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)