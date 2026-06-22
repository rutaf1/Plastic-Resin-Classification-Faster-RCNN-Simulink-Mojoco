imageHeight    = 480;
imageWidth     = 480;
imageChannels  = 3;

numAnchors     = 5;   % rasio [0.25,0.5,1,2,5] sesuai jurnal (Wang et al. 2021)
nmsThreshold   = 0.7;
scoreThreshold = 0.05;
maxProposals   = 100;

numClasses     = 7; % 7 kelas plastik: PET/HDPE/PVC/LDPE/PP/PS/Other
roiPoolSize    = 7;

addpath(fullfile(pwd, 'CustomLayer'));
addpath(fullfile(pwd, 'RegionProposalNetwork'));
addpath(fullfile(pwd, 'RoiPooling'));

net = dlnetwork;

%% Bacbone Renset 50-vd & FPN

tempNet = [
    imageInputLayer([imageHeight imageWidth 3],"Name","imageinput")
    convolution2dLayer([3 3],32,"Name","conv","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],32,"Name","conv_2","Padding","same")
    convolution2dLayer([3 3],64,"Name","conv_1","Padding","same")
    maxPooling2dLayer([3 3],"Name","maxpool","Padding","same","Stride",[2 2])];
net = addLayers(net,tempNet);

% Stage 1 (C2): output 256ch, inner 64ch, spatial 112x112 -> 56x56 after stride
tempNet = [
    convolution2dLayer([1 1],64,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    convolution2dLayer([3 3],64,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    convolution2dLayer([1 1],256,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition")
    reluLayer("Name","relu_2")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],64,"Name","conv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_3")
    convolution2dLayer([3 3],64,"Name","conv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_4")
    convolution2dLayer([1 1],256,"Name","conv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_1")
    reluLayer("Name","relu_5")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],64,"Name","conv_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_6")
    convolution2dLayer([3 3],64,"Name","conv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_7")
    convolution2dLayer([1 1],256,"Name","conv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_2")
    reluLayer("Name","relu_8")];
net = addLayers(net,tempNet);

% Stage 2 (C3): output 512ch, inner 128ch, stride 2 -> spatial 28x28
tempNet = [
    convolution2dLayer([1 1],128,"Name","conv_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_10")
    reluLayer("Name","relu_9")
    DeformableConvolution2DLayer([3 3],128,"Name","deformConv","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_12")
    reluLayer("Name","relu_10")
    convolution2dLayer([1 1],512,"Name","conv_15","Padding","same")
    batchNormalizationLayer("Name","batchnorm_13")];
net = addLayers(net,tempNet);

tempNet = [
    averagePooling2dLayer([2 2],"Name","avgpool2d","Padding","same","Stride",[2 2])
    convolution2dLayer([1 1],512,"Name","conv_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_11")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_3")
    reluLayer("Name","relu_11")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","conv_17","Padding","same")
    batchNormalizationLayer("Name","batchnorm_14")
    reluLayer("Name","relu_12")
    DeformableConvolution2DLayer([3 3],128,"Name","deformConv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15")
    reluLayer("Name","relu_13")
    convolution2dLayer([1 1],512,"Name","conv_16","Padding","same")
    batchNormalizationLayer("Name","batchnorm_16")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_4")
    reluLayer("Name","relu_14")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","conv_19","Padding","same")
    batchNormalizationLayer("Name","batchnorm_17")
    reluLayer("Name","relu_15")
    DeformableConvolution2DLayer([3 3],128,"Name","deformConv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_18")
    reluLayer("Name","relu_16")
    convolution2dLayer([1 1],512,"Name","conv_18","Padding","same")
    batchNormalizationLayer("Name","batchnorm_19")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_5")
    reluLayer("Name","relu_17")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","conv_21","Padding","same")
    batchNormalizationLayer("Name","batchnorm_20")
    reluLayer("Name","relu_18")
    DeformableConvolution2DLayer([3 3],128,"Name","deformConv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_21")
    reluLayer("Name","relu_19")
    convolution2dLayer([1 1],512,"Name","conv_20","Padding","same")
    batchNormalizationLayer("Name","batchnorm_22")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_6")
    reluLayer("Name","relu_20")];
net = addLayers(net,tempNet);

% Stage 3 (C4): output 1024ch, inner 256ch, stride 2 -> spatial 14x14
tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_22","Padding","same")
    batchNormalizationLayer("Name","batchnorm_23")
    reluLayer("Name","relu_21")
    DeformableConvolution2DLayer([3 3],256,"Name","deformConv_4","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_25")
    reluLayer("Name","relu_22")
    convolution2dLayer([1 1],1024,"Name","conv_23","Padding","same")
    batchNormalizationLayer("Name","batchnorm_26")];
net = addLayers(net,tempNet);

tempNet = [
    averagePooling2dLayer([2 2],"Name","avgpool2d_1","Padding","same","Stride",[2 2])
    convolution2dLayer([1 1],1024,"Name","conv_41","Padding","same")
    batchNormalizationLayer("Name","batchnorm_24")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_7")
    reluLayer("Name","relu_23")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_25","Padding","same")
    batchNormalizationLayer("Name","batchnorm_27")
    reluLayer("Name","relu_24")
    DeformableConvolution2DLayer([3 3],256,"Name","deformConv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_28")
    reluLayer("Name","relu_25")
    convolution2dLayer([1 1],1024,"Name","conv_24","Padding","same")
    batchNormalizationLayer("Name","batchnorm_29")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_8")
    reluLayer("Name","relu_26")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_27","Padding","same")
    batchNormalizationLayer("Name","batchnorm_30")
    reluLayer("Name","relu_27")
    DeformableConvolution2DLayer([3 3],256,"Name","deformConv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_31")
    reluLayer("Name","relu_28")
    convolution2dLayer([1 1],1024,"Name","conv_26","Padding","same")
    batchNormalizationLayer("Name","batchnorm_32")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_9")
    reluLayer("Name","relu_29")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_29","Padding","same")
    batchNormalizationLayer("Name","batchnorm_33")
    reluLayer("Name","relu_30")
    DeformableConvolution2DLayer([3 3],256,"Name","deformConv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_34")
    reluLayer("Name","relu_31")
    convolution2dLayer([1 1],1024,"Name","conv_28","Padding","same")
    batchNormalizationLayer("Name","batchnorm_35")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_10")
    reluLayer("Name","relu_32")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_31","Padding","same")
    batchNormalizationLayer("Name","batchnorm_36")
    reluLayer("Name","relu_33")
    DeformableConvolution2DLayer([3 3],256,"Name","deformConv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_37")
    reluLayer("Name","relu_34")
    convolution2dLayer([1 1],1024,"Name","conv_30","Padding","same")
    batchNormalizationLayer("Name","batchnorm_38")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_11")
    reluLayer("Name","relu_35")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_33","Padding","same")
    batchNormalizationLayer("Name","batchnorm_39")
    reluLayer("Name","relu_36")
    DeformableConvolution2DLayer([3 3],256,"Name","deformConv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_40")
    reluLayer("Name","relu_37")
    convolution2dLayer([1 1],1024,"Name","conv_32","Padding","same")
    batchNormalizationLayer("Name","batchnorm_41")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_12")
    reluLayer("Name","relu_38")];
net = addLayers(net,tempNet);

% Stage 4 (C5): output 2048ch, inner 512ch, stride 2 -> spatial 7x7
tempNet = [
    convolution2dLayer([1 1],512,"Name","conv_34","Padding","same")
    batchNormalizationLayer("Name","batchnorm_42")
    reluLayer("Name","relu_39")
    DeformableConvolution2DLayer([3 3],512,"Name","deformConv_10","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_44")
    reluLayer("Name","relu_40")
    convolution2dLayer([1 1],2048,"Name","conv_36","Padding","same")
    batchNormalizationLayer("Name","batchnorm_45")];
net = addLayers(net,tempNet);

tempNet = [
    averagePooling2dLayer([2 2],"Name","avgpool2d_2","Padding","same","Stride",[2 2])
    convolution2dLayer([1 1],2048,"Name","conv_35","Padding","same")
    batchNormalizationLayer("Name","batchnorm_43")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_13")
    reluLayer("Name","relu_41")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],512,"Name","conv_38","Padding","same")
    batchNormalizationLayer("Name","batchnorm_46")
    reluLayer("Name","relu_42")
    DeformableConvolution2DLayer([3 3],512,"Name","deformConv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_47")
    reluLayer("Name","relu_43")
    convolution2dLayer([1 1],2048,"Name","conv_37","Padding","same")
    batchNormalizationLayer("Name","batchnorm_48")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_14")
    reluLayer("Name","relu_44")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],512,"Name","conv_40","Padding","same")
    batchNormalizationLayer("Name","batchnorm_49")
    reluLayer("Name","relu_45")
    DeformableConvolution2DLayer([3 3],512,"Name","deformConv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_50")
    reluLayer("Name","relu_46")
    convolution2dLayer([1 1],2048,"Name","conv_39","Padding","same")
    batchNormalizationLayer("Name","batchnorm_51")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition_15")
    reluLayer("Name","relu_47")
    CoordConv2DLayer([1 1],256,"Name","coordConv_9","Padding","same")
    convolution2dLayer([3 3],512,"Name","conv_51","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_10","Padding","same")];
net = addLayers(net,tempNet);

tempNet = [
    CoordConv2DLayer([1 1],256,"Name","coordConv","Padding","same")
    convolution2dLayer([3 3],512,"Name","conv_42","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_1","Padding","same")];
net = addLayers(net,tempNet);

tempNet = maxPooling2dLayer([5 5],"Name","maxpool_1","Padding","same");
net = addLayers(net,tempNet);

tempNet = maxPooling2dLayer([9 9],"Name","maxpool_2","Padding","same");
net = addLayers(net,tempNet);

tempNet = maxPooling2dLayer([13 13],"Name","maxpool_3","Padding","same");
net = addLayers(net,tempNet);

tempNet = [
    concatenationLayer(3,4,"Name","concat")
    convolution2dLayer([1 1],512,"Name","conv_43","Padding","same")
    batchNormalizationLayer("Name","batchnorm_52")
    convolution2dLayer([3 3],512,"Name","conv_44","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_2","Padding","same")];
net = addLayers(net,tempNet);

tempNet = [
    CoordConv2DLayer([1 1],256,"Name","coordConv_3","Padding","same")
    convolution2dLayer([3 3],512,"Name","conv_45","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_4","Padding","same")];
net = addLayers(net,tempNet);

tempNet = maxPooling2dLayer([5 5],"Name","maxpool_4","Padding","same");
net = addLayers(net,tempNet);

tempNet = maxPooling2dLayer([9 9],"Name","maxpool_5","Padding","same");
net = addLayers(net,tempNet);

tempNet = maxPooling2dLayer([13 13],"Name","maxpool_6","Padding","same");
net = addLayers(net,tempNet);

tempNet = [
    concatenationLayer(3,4,"Name","concat_2")
    convolution2dLayer([1 1],512,"Name","conv_46","Padding","same")
    batchNormalizationLayer("Name","batchnorm_53")
    convolution2dLayer([3 3],512,"Name","conv_47","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_5","Padding","same")];
net = addLayers(net,tempNet);

tempNet = [
    CoordConv2DLayer([1 1],256,"Name","coordConv_6","Padding","same")
    convolution2dLayer([3 3],512,"Name","conv_48","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_7","Padding","same")];
net = addLayers(net,tempNet);

tempNet = maxPooling2dLayer([5 5],"Name","maxpool_7","Padding","same");
net = addLayers(net,tempNet);

tempNet = maxPooling2dLayer([9 9],"Name","maxpool_8","Padding","same");
net = addLayers(net,tempNet);

tempNet = maxPooling2dLayer([13 13],"Name","maxpool_9","Padding","same");
net = addLayers(net,tempNet);

tempNet = [
    concatenationLayer(3,4,"Name","concat_4")
    convolution2dLayer([1 1],512,"Name","conv_49","Padding","same")
    batchNormalizationLayer("Name","batchnorm_54")
    convolution2dLayer([3 3],512,"Name","conv_50","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_8","Padding","same")];
net = addLayers(net,tempNet);

tempNet = maxPooling2dLayer([5 5],"Name","maxpool_10","Padding","same");
net = addLayers(net,tempNet);

tempNet = maxPooling2dLayer([9 9],"Name","maxpool_11","Padding","same");
net = addLayers(net,tempNet);

tempNet = maxPooling2dLayer([13 13],"Name","maxpool_12","Padding","same");
net = addLayers(net,tempNet);

tempNet = [
    concatenationLayer(3,4,"Name","concat_6")
    convolution2dLayer([1 1],512,"Name","conv_52","Padding","same")
    batchNormalizationLayer("Name","batchnorm_55")
    convolution2dLayer([3 3],512,"Name","conv_53","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_11","Padding","same")];
net = addLayers(net,tempNet);

tempNet = resize2dLayer("Name","resize-scale","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","Scale",[2 2]);
net = addLayers(net,tempNet);

tempNet = [
    concatenationLayer(3,2,"Name","concat_1")
    convolution2dLayer([3 3],256,"Name","conv_54","Padding","same")];
net = addLayers(net,tempNet);

tempNet = resize2dLayer("Name","resize-scale_1","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","Scale",[2 2]);
net = addLayers(net,tempNet);

tempNet = [
    concatenationLayer(3,2,"Name","concat_3")
    convolution2dLayer([3 3],256,"Name","conv_55","Padding","same")];
net = addLayers(net,tempNet);

tempNet = resize2dLayer("Name","resize-scale_2","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","Scale",[2 2]);
net = addLayers(net,tempNet);

tempNet = [
    concatenationLayer(3,2,"Name","concat_5")
    convolution2dLayer([3 3],256,"Name","conv_56","Padding","same")];
net = addLayers(net,tempNet);

tempNet = convolution2dLayer([1 1],10,"Name","scoresP2","Padding","same");
net = addLayers(net,tempNet);

tempNet = convolution2dLayer([1 1],20,"Name","boxDeltasP2","Padding","same");
net = addLayers(net,tempNet);

tempNet = convolution2dLayer([1 1],10,"Name","scoresP3","Padding","same");
net = addLayers(net,tempNet);

tempNet = convolution2dLayer([1 1],20,"Name","boxDeltasP3","Padding","same");
net = addLayers(net,tempNet);

tempNet = convolution2dLayer([1 1],10,"Name","scoresP4","Padding","same");
net = addLayers(net,tempNet);

tempNet = convolution2dLayer([1 1],20,"Name","boxDeltasP4","Padding","same");
net = addLayers(net,tempNet);

tempNet = convolution2dLayer([3 3],256,"Name","conv_57","Padding","same");
net = addLayers(net,tempNet);

tempNet = convolution2dLayer([1 1],10,"Name","scoresP5","Padding","same");
net = addLayers(net,tempNet);

tempNet = convolution2dLayer([1 1],20,"Name","boxDeltasP5","Padding","same");
net = addLayers(net,tempNet);

clear tempNet;

net = connectLayers(net,"maxpool","conv_3");
net = connectLayers(net,"maxpool","conv_6");
net = connectLayers(net,"batchnorm_2","addition/in1");
net = connectLayers(net,"batchnorm_3","addition/in2");
net = connectLayers(net,"relu_2","conv_7");
net = connectLayers(net,"relu_2","addition_1/in2");
net = connectLayers(net,"batchnorm_6","addition_1/in1");
net = connectLayers(net,"relu_5","conv_10");
net = connectLayers(net,"relu_5","addition_2/in2");
net = connectLayers(net,"batchnorm_9","addition_2/in1");
net = connectLayers(net,"relu_8","conv_13");
net = connectLayers(net,"relu_8","avgpool2d");
net = connectLayers(net,"relu_8","coordConv");
net = connectLayers(net,"batchnorm_11","addition_3/in2");
net = connectLayers(net,"batchnorm_13","addition_3/in1");
net = connectLayers(net,"relu_11","conv_17");
net = connectLayers(net,"relu_11","addition_4/in2");
net = connectLayers(net,"batchnorm_16","addition_4/in1");
net = connectLayers(net,"relu_14","conv_19");
net = connectLayers(net,"relu_14","addition_5/in2");
net = connectLayers(net,"batchnorm_19","addition_5/in1");
net = connectLayers(net,"relu_17","conv_21");
net = connectLayers(net,"relu_17","addition_6/in2");
net = connectLayers(net,"batchnorm_22","addition_6/in1");
net = connectLayers(net,"relu_20","conv_22");
net = connectLayers(net,"relu_20","avgpool2d_1");
net = connectLayers(net,"relu_20","coordConv_3");
net = connectLayers(net,"batchnorm_26","addition_7/in1");
net = connectLayers(net,"batchnorm_24","addition_7/in2");
net = connectLayers(net,"relu_23","conv_25");
net = connectLayers(net,"relu_23","addition_8/in2");
net = connectLayers(net,"batchnorm_29","addition_8/in1");
net = connectLayers(net,"relu_26","conv_27");
net = connectLayers(net,"relu_26","addition_9/in2");
net = connectLayers(net,"batchnorm_32","addition_9/in1");
net = connectLayers(net,"relu_29","conv_29");
net = connectLayers(net,"relu_29","addition_10/in2");
net = connectLayers(net,"batchnorm_35","addition_10/in1");
net = connectLayers(net,"relu_32","conv_31");
net = connectLayers(net,"relu_32","addition_11/in2");
net = connectLayers(net,"batchnorm_38","addition_11/in1");
net = connectLayers(net,"relu_35","conv_33");
net = connectLayers(net,"relu_35","addition_12/in2");
net = connectLayers(net,"batchnorm_41","addition_12/in1");
net = connectLayers(net,"relu_38","conv_34");
net = connectLayers(net,"relu_38","avgpool2d_2");
net = connectLayers(net,"relu_38","coordConv_6");
net = connectLayers(net,"batchnorm_43","addition_13/in2");
net = connectLayers(net,"batchnorm_45","addition_13/in1");
net = connectLayers(net,"relu_41","conv_38");
net = connectLayers(net,"relu_41","addition_14/in2");
net = connectLayers(net,"batchnorm_48","addition_14/in1");
net = connectLayers(net,"relu_44","conv_40");
net = connectLayers(net,"relu_44","addition_15/in2");
net = connectLayers(net,"batchnorm_51","addition_15/in1");
net = connectLayers(net,"coordConv_1","maxpool_1");
net = connectLayers(net,"coordConv_1","maxpool_2");
net = connectLayers(net,"coordConv_1","maxpool_3");
net = connectLayers(net,"coordConv_1","concat/in4");
net = connectLayers(net,"maxpool_1","concat/in2");
net = connectLayers(net,"maxpool_2","concat/in1");
net = connectLayers(net,"maxpool_3","concat/in3");
net = connectLayers(net,"coordConv_2","concat_1/in1");
net = connectLayers(net,"coordConv_4","maxpool_4");
net = connectLayers(net,"coordConv_4","maxpool_5");
net = connectLayers(net,"coordConv_4","maxpool_6");
net = connectLayers(net,"coordConv_4","concat_2/in4");
net = connectLayers(net,"maxpool_4","concat_2/in2");
net = connectLayers(net,"maxpool_5","concat_2/in1");
net = connectLayers(net,"maxpool_6","concat_2/in3");
net = connectLayers(net,"coordConv_5","resize-scale");
net = connectLayers(net,"coordConv_5","concat_3/in1");
net = connectLayers(net,"coordConv_7","maxpool_7");
net = connectLayers(net,"coordConv_7","maxpool_8");
net = connectLayers(net,"coordConv_7","maxpool_9");
net = connectLayers(net,"coordConv_7","concat_4/in4");
net = connectLayers(net,"maxpool_7","concat_4/in2");
net = connectLayers(net,"maxpool_8","concat_4/in1");
net = connectLayers(net,"maxpool_9","concat_4/in3");
net = connectLayers(net,"coordConv_8","resize-scale_1");
net = connectLayers(net,"coordConv_8","concat_5/in1");
net = connectLayers(net,"coordConv_10","maxpool_10");
net = connectLayers(net,"coordConv_10","maxpool_11");
net = connectLayers(net,"coordConv_10","maxpool_12");
net = connectLayers(net,"coordConv_10","concat_6/in4");
net = connectLayers(net,"maxpool_10","concat_6/in2");
net = connectLayers(net,"maxpool_11","concat_6/in1");
net = connectLayers(net,"maxpool_12","concat_6/in3");
net = connectLayers(net,"coordConv_11","resize-scale_2");
net = connectLayers(net,"coordConv_11","conv_57");
net = connectLayers(net,"resize-scale","concat_1/in2");
net = connectLayers(net,"resize-scale_1","concat_3/in2");
net = connectLayers(net,"resize-scale_2","concat_5/in2");
net = connectLayers(net,"conv_54","scoresP2");
net = connectLayers(net,"conv_54","boxDeltasP2");
net = connectLayers(net,"conv_55","scoresP3");
net = connectLayers(net,"conv_55","boxDeltasP3");
net = connectLayers(net,"conv_56","scoresP4");
net = connectLayers(net,"conv_56","boxDeltasP4");
net = connectLayers(net,"conv_57","scoresP5");
net = connectLayers(net,"conv_57","boxDeltasP5");
net = initialize(net);

%% Region Proposal Network
image = dlarray(rand(imageHeight, imageWidth, imageChannels, 1, 'single'), 'SSCB');

outputNames = {'scoresP2', 'boxDeltasP2', ...
               'scoresP3', 'boxDeltasP3', ...
               'scoresP4', 'boxDeltasP4', ...
               'scoresP5', 'boxDeltasP5', ...
               'conv_54',  'conv_55', 'conv_56', 'coordConv_11'};

netOutputs = cell(1, numel(outputNames));
[netOutputs{:}] = predict(net, image, 'Outputs', outputNames);

scoresP2    = netOutputs{1};
boxDeltasP2 = netOutputs{2};
scoresP3    = netOutputs{3};
boxDeltasP3 = netOutputs{4};
scoresP4    = netOutputs{5};
boxDeltasP4 = netOutputs{6};
scoresP5    = netOutputs{7};
boxDeltasP5 = netOutputs{8};
fmP2        = netOutputs{9};
fmP3        = netOutputs{10};
fmP4        = netOutputs{11};
fmP5        = netOutputs{12};

rpnOutputs = {
    extractdata(scoresP2),  extractdata(boxDeltasP2), ...
    extractdata(scoresP3),  extractdata(boxDeltasP3), ...
    extractdata(scoresP4),  extractdata(boxDeltasP4), ...
    extractdata(scoresP5),  extractdata(boxDeltasP5)
};

anchorBoxes = generateAnchorBoxes();

imageSize = [imageHeight, imageWidth];
[allProposals, allScores] = processRPNOutputs(...
    rpnOutputs, anchorBoxes, imageSize, ...
    'NMSThreshold', nmsThreshold, ...
    'ScoreThreshold', scoreThreshold, ...
    'MaxProposals', maxProposals);

fprintf('Total proposals: %d\n', size(allProposals, 1));

%% ROI Pooling
featureMaps = {
    squeeze(extractdata(fmP2)), ...
    squeeze(extractdata(fmP3)), ...
    squeeze(extractdata(fmP4)), ...
    squeeze(extractdata(fmP5))
};
numFMChannels = size(featureMaps{1}, 3);
roiFeatures   = roiPooling(featureMaps, allProposals, imageSize, roiPoolSize);

%% Fully Connected Layer
fcNet = dlnetwork;

% FC head: flatten the 7x7x256 RoI directly (NO global-average-pool, which
% would discard the spatial info RoI pooling produced).
tempNet = [
    imageInputLayer([roiPoolSize roiPoolSize numFMChannels], "Name", "roiInput", "Normalization", "none")
    fullyConnectedLayer(1024, "Name", "fc1")
    reluLayer("Name", "relu_fc1")
    fullyConnectedLayer(1024, "Name", "fc2")
    reluLayer("Name", "relu_fc2")];
fcNet = addLayers(fcNet, tempNet);

% Classifier includes a background class (numClasses+1), matching Training.m's
% BG_CLASS = numClasses+1 convention.
tempNet = [
    fullyConnectedLayer(numClasses + 1, "Name", "fc3_cls")
    softmaxLayer("Name", "classScores")];
fcNet = addLayers(fcNet, tempNet);

% Per-foreground-class bbox regression (numClasses*4), matching Training.m's
% gpuSampleROIs/gpuDetLoss target layout. Background has no bbox regression.
tempNet = fullyConnectedLayer(numClasses * 4, "Name", "fc3_bbox");
fcNet = addLayers(fcNet, tempNet);

clear tempNet;

fcNet = connectLayers(fcNet, "relu_fc2", "fc3_cls");
fcNet = connectLayers(fcNet, "relu_fc2", "fc3_bbox");
fcNet = initialize(fcNet);

%% Forward Pass FC Head
classScores = [];
bboxPreds   = [];

for i = 1:size(roiFeatures, 4)
    roi = dlarray(roiFeatures(:,:,:,i), 'SSCB');
    [cls, bbox] = predict(fcNet, roi, ...
        'Outputs', {'classScores', 'fc3_bbox'});
    classScores = [classScores; extractdata(cls)'];
    bboxPreds   = [bboxPreds;   extractdata(bbox)'];
end

fprintf('Total detections: %d\n', size(classScores, 1));
fprintf('Class scores size: %dx%d\n', size(classScores));
fprintf('BBox predictions size: %dx%d\n', size(bboxPreds));
