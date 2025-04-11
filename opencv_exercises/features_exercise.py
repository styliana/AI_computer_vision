'''This is an exercise to make you get comfortable in using features detectors and extractor for performing
different tasks. In this exercise, you are going to create an object classificator (cool!), but without using any
machine or deep learning methods (less cool). As the first exercise, you will not be evaluated. The
following are the steps that you should perform to achieve the final result.

1. Pick some items you want to classify. I suggest for starting to pick 3 simple recognizable objects.
For the sake of the example, I will use books. Be sure to have such items at home.

2. Download from the internet the cover for each book you have picked. These will be your database.

3. In order to perform the classification, you need to extract the features from each book cover. To do
so, choose a feature extractor among SIFT, ORB, or A-KAZE, extract the descriptor from each book
cover, and store them in a list.

4. Now that the database is ready, you can provide some images, i.e. the test images, to your program
to check if it correctly classifies the objects. You can provide the object, i.e. the book, in two ways: in
real-time, by using the webcam of your pc, or by taking a picture of your object with your camera
and then load it from within the program. It is up to you!

5. Once that your test image has been loaded, you need to match it with the descriptors database you
created before. To keep track of the matches, create a new list and store the matches inside it.

6. Once that you have the list of matches (be sure to take only good matches ;D), you can pick the
index of the maximum value. That will be the class of the object!

7. To make the things a little bit more robust, you can check if the maximum value is greater than a
threshold you define.'''

#coming soon