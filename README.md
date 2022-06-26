# Deep Cross Network Product Recommender

The following is a repo for my research work into deep cross network recommender systems using millions of datapoints for books ratings that I collected.

Deep cross networks allow for features to interact before they are fed into a DNN. Beyong the user id, book title and user rating, engineered features including the user's favourite genres and the books primary genre classification were devised to give the DCN a richer reference for feature interaction.

The model's goal is to predict the user's preference for a given book we wish to recommend to the user. Inference is straight forward required identifying a user to recommend a book to, identifying the book we wish to recommend, and the engineered features that are fixed according to user and book respectively - as is demonstrated towards the end of the research notebook.
