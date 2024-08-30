In this a Pixel to Pixel GAN has been trained and tested.

The generate_dataset/ folder contains a modified version of the repository that implements a deterministic reverse parking path finder. This has got an average inference timeof 1.95 secs.
(Link to the repo: https://github.com/Pandas-Team/Automatic-Parking)

Whereas, after training the GAN, we are able to acheive average inference time of 0.39 secs. Which clearly shows the reduction in inference time by almost half.

(Also note that with more complex scenarios, the time required to layout paths increases greatly in most of the deterministic methods, whereas it almost remains constant in NN based methods.)
