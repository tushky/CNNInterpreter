# CNNInterpreter
### Collection of populer Convolutional Neural Network (CNN) interpretation methods useful for debugging and understanding predictions of CNN.

## This package contains implementation of following methods

1. Decovlolution Network
2. Guided Backpropogation
3. Salienecy Map
4. Class Specific Saliency Map
5. Class Activation Map (CAM)
6. Gradient weighted Class Activation Map (Grad-CAM)
7. Gradient weighted Class Activation Map ++ (Grad-CAM++)
8. Score weighted Class Activation Map(Score-CAM)
9. Guided CAM, Grad-CAM, Grad-CAM++, Score-CAM
10. Deep Dream


## Class Activation Map Based Methods

<table border=0 >
	<tbody>
    <tr>
            <td align="center"> Image </td>
			<td align="center"> VGG16 </td>
			<td align="center"> Resnet34 </td>
			<td align="center"> GoogleNet</td>
            <td align="center"> SuffleNet</td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/birds.png"> </td>
			<td width="20%"> <img src="./results/birds_gradcam_vgg16.png"> </td>
			<td width="20%"> <img src="./results/birds_gradcam_resnet34.png"> </td>
			<td width="20%"> <img src="./results/birds_gradcam_googlenet.png"> </td>
			<td width="20%"> <img src="./results/birds_gradcam_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/church.png"> </td>
			<td width="20%"> <img src="./results/church_gradcam_vgg16.png"> </td>
			<td width="20%"> <img src="./results/church_gradcam_resnet34.png"> </td>
			<td width="20%"> <img src="./results/church_gradcam_googlenet.png"> </td>
			<td width="20%"> <img src="./results/church_gradcam_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/spider.png"> </td>
			<td width="20%"> <img src="./results/spider_gradcam_vgg16.png"> </td>
			<td width="20%"> <img src="./results/spider_gradcam_resnet34.png"> </td>
			<td width="20%"> <img src="./results/spider_gradcam_googlenet.png"> </td>
			<td width="20%"> <img src="./results/spider_gradcam_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/water-bird.png"> </td>
			<td width="20%"> <img src="./results/water-bird_gradcam_vgg16.png"> </td>
			<td width="20%"> <img src="./results/water-bird_gradcam_resnet34.png"> </td>
			<td width="20%"> <img src="./results/water-bird_gradcam_googlenet.png"> </td>
			<td width="20%"> <img src="./results/water-bird_gradcam_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/clock.png"> </td>
			<td width="20%"> <img src="./results/clock_gradcam_vgg16.png"> </td>
			<td width="20%"> <img src="./results/clock_gradcam_resnet34.png"> </td>
			<td width="20%"> <img src="./results/clock_gradcam_googlenet.png"> </td>
			<td width="20%"> <img src="./results/clock_gradcam_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/cat_dog.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_gradcam_vgg16.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_gradcam_resnet34.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_gradcam_googlenet.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_gradcam_shufflenet.png"> </td>
		</tr>
	</tbody>
</table>

<table border=0 >
	<tbody>
    <tr>
            <td align="center"> Original </td>
			<td align="center"> CAM </td>
			<td align="center"> Grad-CAM </td>
			<td align="center"> Grad-CAM++</td>
            <td align="center"> Score-CAM</td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/birds.png"> </td>
			<td width="20%"> <img src="./results/birds_cam.png"> </td>
			<td width="20%"> <img src="./results/birds_gradcam.png"> </td>
			<td width="20%"> <img src="./results/birds_gradcam++.png"> </td>
            <td width="20%"> <img src="./results/birds_scorecam.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/church.png"> </td>
			<td width="20%"> <img src="./results/church_cam.png"> </td>
			<td width="20%"> <img src="./results/church_gradcam.png"> </td>
			<td width="20%"> <img src="./results/church_gradcam++.png"> </td>
            <td width="20%"> <img src="./results/church_scorecam.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/spider.png"> </td>
			<td width="20%"> <img src="./results/spider_cam.png"> </td>
			<td width="20%"> <img src="./results/spider_gradcam.png"> </td>
			<td width="20%"> <img src="./results/spider_gradcam++.png"> </td>
            <td width="20%"> <img src="./results/spider_scorecam.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/water-bird.png"> </td>
			<td width="20%"> <img src="./results/water-bird_cam.png"> </td>
			<td width="20%"> <img src="./results/water-bird_gradcam.png"> </td>
			<td width="20%"> <img src="./results/water-bird_gradcam++.png"> </td>
            <td width="20%"> <img src="./results/water-bird_scorecam.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/clock.png"> </td>
			<td width="20%"> <img src="./results/clock_cam.png"> </td>
			<td width="20%"> <img src="./results/clock_gradcam.png"> </td>
			<td width="20%"> <img src="./results/clock_gradcam++.png"> </td>
            <td width="20%"> <img src="./results/clock_scorecam.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/cat_dog.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_cam.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_gradcam.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_gradcam++.png"> </td>
            <td width="20%"> <img src="./results/cat_dog_scorecam.png"> </td>
		</tr>
	</tbody>
</table>

## Sensitivity Maps Based Methods

### Integrated Gradients

<table border=0 >
	<tbody>
    <tr>
            <td align="center"> Image </td>
			<td align="center"> VGG16 </td>
			<td align="center"> Resnet34 </td>
			<td align="center"> GoogleNet</td>
            <td align="center"> SuffleNet</td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/birds.png"> </td>
			<td width="20%"> <img src="./results/birds_integrated_vgg16.png"> </td>
			<td width="20%"> <img src="./results/birds_integrated_resnet34.png"> </td>
			<td width="20%"> <img src="./results/birds_integrated_googlenet.png"> </td>
			<td width="20%"> <img src="./results/birds_integrated_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/church.png"> </td>
			<td width="20%"> <img src="./results/church_integrated_vgg16.png"> </td>
			<td width="20%"> <img src="./results/church_integrated_resnet34.png"> </td>
			<td width="20%"> <img src="./results/church_integrated_googlenet.png"> </td>
			<td width="20%"> <img src="./results/church_integrated_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/spider.png"> </td>
			<td width="20%"> <img src="./results/spider_integrated_vgg16.png"> </td>
			<td width="20%"> <img src="./results/spider_integrated_resnet34.png"> </td>
			<td width="20%"> <img src="./results/spider_integrated_googlenet.png"> </td>
			<td width="20%"> <img src="./results/spider_integrated_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/water-bird.png"> </td>
			<td width="20%"> <img src="./results/water-bird_integrated_vgg16.png"> </td>
			<td width="20%"> <img src="./results/water-bird_integrated_resnet34.png"> </td>
			<td width="20%"> <img src="./results/water-bird_integrated_googlenet.png"> </td>
			<td width="20%"> <img src="./results/water-bird_integrated_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/clock.png"> </td>
			<td width="20%"> <img src="./results/clock_integrated_vgg16.png"> </td>
			<td width="20%"> <img src="./results/clock_integrated_resnet34.png"> </td>
			<td width="20%"> <img src="./results/clock_integrated_googlenet.png"> </td>
			<td width="20%"> <img src="./results/clock_integrated_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/cat_dog.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_integrated_vgg16.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_integrated_resnet34.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_integrated_googlenet.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_integrated_shufflenet.png"> </td>
		</tr>
	</tbody>
</table>


### Smooth Gradients (SmoothGrad)

<table border=0 >
	<tbody>
    <tr>
            <td align="center"> Image </td>
			<td align="center"> VGG16 </td>
			<td align="center"> Resnet34 </td>
			<td align="center"> GoogleNet</td>
            <td align="center"> SuffleNet</td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/birds.png"> </td>
			<td width="20%"> <img src="./results/birds_smooth_vgg16.png"> </td>
			<td width="20%"> <img src="./results/birds_smooth_resnet34.png"> </td>
			<td width="20%"> <img src="./results/birds_smooth_googlenet.png"> </td>
			<td width="20%"> <img src="./results/birds_smooth_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/church.png"> </td>
			<td width="20%"> <img src="./results/church_smooth_vgg16.png"> </td>
			<td width="20%"> <img src="./results/church_smooth_resnet34.png"> </td>
			<td width="20%"> <img src="./results/church_smooth_googlenet.png"> </td>
			<td width="20%"> <img src="./results/church_smooth_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/spider.png"> </td>
			<td width="20%"> <img src="./results/spider_smooth_vgg16.png"> </td>
			<td width="20%"> <img src="./results/spider_smooth_resnet34.png"> </td>
			<td width="20%"> <img src="./results/spider_smooth_googlenet.png"> </td>
			<td width="20%"> <img src="./results/spider_smooth_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/water-bird.png"> </td>
			<td width="20%"> <img src="./results/water-bird_smooth_vgg16.png"> </td>
			<td width="20%"> <img src="./results/water-bird_smooth_resnet34.png"> </td>
			<td width="20%"> <img src="./results/water-bird_smooth_googlenet.png"> </td>
			<td width="20%"> <img src="./results/water-bird_smooth_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/clock.png"> </td>
			<td width="20%"> <img src="./results/clock_smooth_vgg16.png"> </td>
			<td width="20%"> <img src="./results/clock_smooth_resnet34.png"> </td>
			<td width="20%"> <img src="./results/clock_smooth_googlenet.png"> </td>
			<td width="20%"> <img src="./results/clock_smooth_shufflenet.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/cat_dog.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_smooth_vgg16.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_smooth_resnet34.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_smooth_googlenet.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_smooth_shufflenet.png"> </td>
		</tr>
	</tbody>
</table>

## Deconvolution Network Based Methods

### Guided Backpropogation

<table border=0 >
	<tbody>
    <tr>
            <td align="center"> Image </td>
			<td align="center"> AlexNet </td>
			<td align="center"> VGG13 </td>
			<td align="center"> VGG16 </td>
            <td align="center"> VGG19 </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/birds.png"> </td>
			<td width="20%"> <img src="./results/birds_True_alexnet.png"> </td>
			<td width="20%"> <img src="./results/birds_True_vgg13.png"> </td>
			<td width="20%"> <img src="./results/birds_True_vgg16.png"> </td>
			<td width="20%"> <img src="./results/birds_True_vgg19.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/church.png"> </td>
			<td width="20%"> <img src="./results/church_True_alexnet.png"> </td>
			<td width="20%"> <img src="./results/church_True_vgg13.png"> </td>
			<td width="20%"> <img src="./results/church_True_vgg16.png"> </td>
			<td width="20%"> <img src="./results/church_True_vgg19.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/spider.png"> </td>
			<td width="20%"> <img src="./results/spider_True_alexnet.png"> </td>
			<td width="20%"> <img src="./results/spider_True_vgg13.png"> </td>
			<td width="20%"> <img src="./results/spider_True_vgg16.png"> </td>
			<td width="20%"> <img src="./results/spider_True_vgg19.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/water-bird.png"> </td>
			<td width="20%"> <img src="./results/water-bird_True_alexnet.png"> </td>
			<td width="20%"> <img src="./results/water-bird_True_vgg13.png"> </td>
			<td width="20%"> <img src="./results/water-bird_True_vgg16.png"> </td>
			<td width="20%"> <img src="./results/water-bird_True_vgg19.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/clock.png"> </td>
			<td width="20%"> <img src="./results/clock_True_alexnet.png"> </td>
			<td width="20%"> <img src="./results/clock_True_vgg13.png"> </td>
			<td width="20%"> <img src="./results/clock_True_vgg16.png"> </td>
			<td width="20%"> <img src="./results/clock_True_vgg19.png"> </td>
		</tr>
		<tr>
            <td width="20%"> <img src="./results/cat_dog.png"> </td>
			<td width="20%"> <img src="./results/cat_True_alexnet.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_True_vgg13.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_True_vgg16.png"> </td>
			<td width="20%"> <img src="./results/cat_dog_True_vgg19.png"> </td>
		</tr>
	</tbody>
</table>