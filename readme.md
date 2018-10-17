### Aproach on the Text Prediction
Build a 2 layer GRU with an encoder-decoder architecture with the sentence reversed. Building an extra layer over the GRU helps it capture knowledge representations better over time. GRU cell was chosen instead of the LSTM simply for performance execution and hence the similar results the Occam's razor principle applies. The encoder capture the sequence of strokes that generated the sentence and the decoder takes this dense representation of the sequence and unroll it in the text prediction.

Since this encoder-decoder correspond to a many-to-many application I decided to pad both input and output at x.mean() + 2* x/std() to normalize the sentences lenght with zero padding. This value covered over 95% of both strokes and sentences lenghts in the dataset.

The text was also one-hot encoded with different representations for lower and upper chars as well as symbols. The char tokenizer was built with the help of the keras Tokenizer library.

### Improvements Scratch
This task could be tackled as image recognition task instead of sequential using the generated stroke as input for convolutions with objetives similar to yolo (detect position and bounding boxes) and that could be further convoluted to predict the words. This doesn't seem a good approach since many of the samples varies greatly in stroke style and lenght. A similar approach could be used to generate the strokes but, again, it doesn't seem like a good approach to generate realistic handwriting strokes given the GANs outputs in famous applications as pix2pix.

The task could be also tackled as CTC for the text classification but since the GRU with fixed size worked well enough there was no interest in this approach as well.

The addition of attention layers on the models could also improve its performance, specially given longer sequences. 

For the handwriting genearation my first attempt was to use an autoencoder which would receive the text, generate a latent representation and the output it to text again. My original idea was to remove the decoder and use a different sequence model to output the X, Y and Pen values to generate the strokes. This approach didn't prove useful given the nature of the strokes.

I also tried to assume that the hipothesis that each letter could be windowed by a smaller sequence of strokes could be true. But for the simplest case the window didn't capture a whole stroke given that many cursive words have a long stroke for a whole sentence invalidating my approach. And again, given the nature of the data distribution it didn't work well.

I recently heard about variational autoencoders and given the Alex paper instructions I believe both approaches would result in similar results. I tried to implement Alex solution but I: had a hard time on the MDN (Mixture Density Network) mostly because given the absence of a implemented layer in the frameworks I use most (keras, tensorflow) and the nature of the test I had to implement it myself from scratch. But up to this point I still haven't been able to make it work properly.

### Setup Used
Keras                             2.2.4             
Keras-Applications                1.0.6             
Keras-Preprocessing               1.0.5
numpy                             1.15.2
tb-nightly                        1.12.0a20181015   
tensorboard                       1.11.0            
tensorflow                        1.12.0rc0         
tensorflow-probability            0.4.0
tf-nightly                        1.12.0.dev20181012
tf-nightly-gpu                    1.12.0.dev20181012
tfp-nightly                       0.5.0.dev20181015 
tfp-nightly-gpu                   0.5.0.dev20181015 


## References:
 [1] Alex Graves paper - https://arxiv.org/abs/1308.0850
