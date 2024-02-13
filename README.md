# Audio to Image using various models
Recently I have been interested in audio-to-image generation. That is using the audio of something, let's say the sound of being inside a beach to generate an image of a beach. Now I didn't find a dataset that pairs audio to images, so I made one based on [this paper](https://arxiv.org/pdf/2109.13354.pdf). Basically an MNSIT-FSDD dataset. I also made branches for each step of the process while I was trying to figure out things. First was WGAN, I used a [YouTube tutorial](https://youtu.be/pG0QZ7OddX4?si=brUfAghFf2_xbcc6) for that. Then I went on to learn about and code up a VAE (specifically a CVAE) with conv layers. I also coded up a VAEGAN and used the loss from the first paper I linked (I am not still sure it's correct lol). Done with my beloved, Python.

## Results after 100 epochs
![CVAE](https://github.com/ZeroMeOut/ImageFromAudio/assets/63326326/6c321739-d06e-408a-bed9-aaf884d6b6ac) 
|:--:| 
| *CVAE* |

![VAEGAN](https://github.com/ZeroMeOut/ImageFromAudio/assets/63326326/5222ea66-a78c-4c2f-a19b-87f2750820fb)
|:--:| 
| *VAEGAN* |

I know I am supposed to show some charts for the loss, I will probably do that later (or you can help me with that :)).

## Thoughts
It seems to struggle to reconstruct the image after a certain point. I honestly don't know if it's because of the loss function I use, the learning rate, the one-to-one pairing for making the dataset, or some other thing I can't think of. But I will figure it out eventually, this was just an attempt after all. Feel free to yoink it and do whatever :) orrr help figure it out idk just a thought. 
