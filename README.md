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
            <td align="center"> Original </td>
			<td align="center"> CAM </td>
			<td align="center"> Grad-CAM </td>
			<td align="center"> Grad-CAM++</td>
            <td align="center"> Score-CAM</td>
		</tr>
		<tr>
            <td width="20%"> <img src="./data/dogs.jpg"> </td>
			<td width="20%"> <img src="./data/resnet34_dogs_cam.png"> </td>
			<td width="20%"> <img src="./data/resnet34_dogs_gradcam.png"> </td>
			<td width="20%"> <img src="./data/resnet34_dogs_gradcam++.png"> </td>
            <td width="20%"> <img src="./data/resnet34_dogs_scorecam.png"> </td>
		</tr>
        <tr>
            <td width="20%"> <img src="./data/cat_dog.png"> </td>
			<td width="20%"> <img src="./data/resnet34_cat_dog_cam.png"> </td>
			<td width="20%"> <img src="./data/resnet34_cat_dog_gradcam.png"> </td>
			<td width="20%"> <img src="./data/resnet34_cat_dog_gradcam++.png"> </td>
            <td width="20%"> <img src="./data/resnet34_cat_dog_scorecam.png"> </td>
		</tr>
        <tr>
            <td width="20%"> <img src="./data/clock.jpg"> </td>
			<td width="20%"> <img src="./data/resnet34_clock_cam.png"> </td>
			<td width="20%"> <img src="./data/resnet34_clock_gradcam.png"> </td>
			<td width="20%"> <img src="./data/resnet34_clock_gradcam++.png"> </td>
            <td width="20%"> <img src="./data/resnet34_clock_scorecam.png"> </td>
		</tr>
	</tbody>
</table>

## Deconvolution Network Based Methods

1. DeconvNet
2. Guided Backpropogation

<table border=0 >
	<tbody>
    <tr>
			<td width="10%"> Layer Number </td>
            <td align="center"> Top 5 Filters </td>
		</tr>
		<tr>
			<td width="10%"> 10 </td>
            <td width="90%"> <img src="./data/clock_deconv_output_layer_10.png"> </td>
		</tr>
        <tr>
            <td width="10%"> 15 </td>
            <td width="90%"> <img src="./data/clock_deconv_output_layer_15.png"> </td>
		</tr>
		<tr>
           <td width="10%"> 20 </td>
            <td width="90%"> <img src="./data/clock_deconv_output_layer_20.png"> </td>
		</tr>
		<tr>
            <td width="10%"> 25 </td>
            <td width="90%"> <img src="./data/clock_deconv_output_layer_25.png"> </td>
		</tr>
		<tr>
            <td width="10%"> 28 </td>
            <td width="90%"> <img src="./data/clock_deconv_output_layer_28.png"> </td>
		</tr>
	</tbody>
</table>
