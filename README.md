
High level story:

Data Collection

* Recording training data - started with two laps forward and backwards (to balance out the left and right turns).  Found car hard to control using keyboard but managed to drive better using xbox controller.  The xbox controller on a mac can be used by running this software.

* Added in recovery data - spent a time manually cleaning the data - removing frames related as I drove towards the edge of the track.  I did this by deleting images that related to the drive from the center to the edge of the road.  

AWS 

* Wanted to use AWS to access GPU's to speed up training.  Error in the instance name in docs - needed to use this one!  In total for this project I used 44 hours of AWS compute over several weeks.

Initial Experiemnts


* I was initially working with full images (center, left and right) - but could not get past bridge - even when adding in more training data.  Initially was using 15000 images, with additional laps 24000 images.  Model architecture was .... just say it was final one with varying size in first layer.  https://keras.io/visualization/
* Tried removing top and bottom of images and rescaling to half size - still could not get past bridge.
* Tried recording bridge approach multiple times and adding that to training data - this was time consuming and not particulalry effective - not helping me get across bridge.  I also did not find it attractive to be patching model with new data - hard to reproduce exactly what I did.
* Experiments with early stopping were also unsatisfactory - although they did require me to dig about in the keras documentation to find out how to do early stopping using generators on both the train and validation step.  You can see this code in the github repo for my intial experiments.  
* In the end decided it was better to train with dropout and just test the model on the track - if the model worked then great!
* Initial experiments are captured in this git repo.

Reboot and Success!

* At this point I decided to check out the Slack channel!  I had previously been hanging out on the forums - since I did not realise that there was a project specific slack channel.  I found some great appraoches by () and decided to also adopt the image augmentation approach - choosing to change brightness, translation and flipping, plus added 3 1x1 filters in model.  This got me reliably over the bridge but the first sharp left hand bend proved elusive.  
* Tried experiments where I trained on all the data then downsampled the small angles - but this felt less reproducible - and did not reliably help me get round the bend after the bridge.
* Decided to spend more time looking at the distribution of the training data - in the end wrote generator to return stratified balanced data. I decided to bin data based on angles and sampled uniformly in generator across the bins.  This was the breakthrough - getting me all the way round the track!  Show plots of generator versus raw.
* Show generator and example images.
* Final architecture - how many parameters?  https://keras.io/visualization/
* With this model I could have a large number of batches per epoch - 400 images x 10k batches = 400k images per epoch!  I trained in 5 epoch steps - took <2 mins on AWS.  After 50 epochs I reliably had a model that would drive the track.  The sharp left hand bend is something I still need to work on since the car does make a pretty violent recovery - I suspect that this could be due to some of the selected hyperparameters for the translated images and left and right images.  At this point I ran out of time.
* The model also got round most of the second track - only failing on the hill - I suspect that this is due to trimming the images too aggressively.
* Include video on youtube.

Lessons Learnt

* MSE not particularly helpful - nor was early stopping.  Heavy dropout seemed more effective and image augmentation.  This is a form a regularisation - just surprised me how early stopping (my go to regularisation technique) did not help - I think partly because the training data was small relative to the model parameters - so high variance to training/ validation randomisation.  Maybe with augmentation will be a lot better.
* Always understand the ditribution of the data.  Initially I did not plot the distribution of the angles - I shoudl have gone straight to the data and realised that data was unbalanced.  I was focussing too much on forward and backwards training generating balance - but the problem was the under represented recovery data - in hindsight this seems obvious but it wasted several rounds of experiments over a number of weeks!
* I was trying to avoid image augmentation since it introduced yet more hyperparameters - for example the range of brightness, translation.  However this was the break through for me and got me reliably over the bridge. The data proved the most important thing - and understanding the data.  This is the first time I have been in a position to generate more training data - in my previous applications of machine learnign I have a fixed data set - often collected at great expense. It was an odd feeling to be able to cheaply create more data.  A great lesson from this project.
* I was also at time daunted by the number of hyper-parameters.  For example the model architecture, learning rate, image augmentation parameters, amount of training data, test/ valid split.  In the end I had to go with my gut feel - lifting an architecture from this paper and just fixing my learning rate using the Adam optimizer.  Turns out that so long as these are sensible that they are second order to the data itself.   


```python

```


```python

```


```python

```


```python

```

### Rubric

#### Quality of Code
Criteria 	Meets Specifications

Is the code functional?
	
The model provided can be used to successfully operate the simulation.

Is the code usable and readable?
	
The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.

#### Model Architecture and Training Strategy

Criteria 	Meets Specifications

Has an appropriate model architecture been employed for the task?
	
The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model.

Has an attempt been made to reduce overfitting of the model?

Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.

Have the model parameters been tuned appropriately?

Learning rate parameters are chosen with explanation, or an Adam optimizer is used.

Is the training data chosen appropriately?
	
Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).

#### Architecture and Training Documentation

Criteria 	Meets Specifications

Is the solution design documented?
	
The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

Is the model architecture documented?
	
The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

Is the creation of the training dataset and training process documented?
	
The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset should be included.

#### Simulation

Criteria 	Meets Specifications

Is the car able to navigate correctly on test data?
	
No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).
