# AI-by-design
Using Machine Learning to classify attributes of a logo

## Overview

This program is broken into two parts:

* A program in Go to grab logo images from Dribbble and sort them by their 'tag' attributes
* A program using the Tensorflow framework to train a network that classifies logos by their attributes

## Getting data from Dribble

Getting data from the go program is simple enough. Just pull the repo into your Gopath, add a config.json file that looks like:

```
{
  "dribbbleKey": "MY_DRIBBLE_KEY"
}
```

You'll also want to create several directories for storing the logo assets:

```
/tmp
/tmp-bw
/tmp-cifar
/tmp-cifar-sm
```
The tmp-cifar directories store the data in a `cifar` format, which is a byte array that has the 'label' of the image as the first byte, and then the image data as the rest of the bytes. The cifar program should transform these automatically.

After creating these directories, just run `go run main.go`, and it should pull the logos.

## Running the neural network

The neural network is set up to run in a Docker container, although it can also be run on the host machine CPU or GPU depending on set up.

In order to train the nn using the pulled logo data, first build the Docker image:

`docker build -t ai_by_design .`

Then run the container:

`docker run -i -t ai_by_design`
