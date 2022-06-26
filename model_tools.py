import tensorflow as tf
import tensorflow_recommenders as tfrs

class RankingModel(tf.keras.Model):

    def __init__(self, book_title_stringlookup, user_id_stringlookup):
        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([user_id_stringlookup,
          tf.keras.layers.Embedding(user_id_stringlookup.vocabulary_size()+1, embedding_dimension)
        ])

        # Compute embeddings for books.
        self.book_embeddings = tf.keras.Sequential([book_title_stringlookup,
          tf.keras.layers.Embedding(book_title_stringlookup.vocabulary_size()+1, embedding_dimension)
        ])
        
        # Compute predictions.
        self.ratings = tf.keras.Sequential([
          tf.keras.layers.Dense(32, activation="relu"),
          tf.keras.layers.Dense(16, activation="relu"),
          tf.keras.layers.Dense(8, activation="relu"),
          tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):

        user_id, book_title = inputs

        user_embedding = self.user_embeddings(user_id)
        book_embedding = self.book_embeddings(book_title)

        return self.ratings(tf.concat([user_embedding, book_embedding], axis=1))
        
        
        
class BookRecommenderModel(tfrs.models.Model):

    def __init__(self, book_title_stringlookup, user_id_stringlookup):
        super().__init__(self)
        self.ranking_model: tf.keras.Model = RankingModel(book_title_stringlookup, user_id_stringlookup)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
          loss = tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features: dict) -> tf.Tensor:
        return self.ranking_model(
            (features[:, 1], features[:, 0]))

    def compute_loss(self, features: dict, training=False) -> tf.Tensor:

        labels =  tf.strings.to_number(features[:,2])

        rating_predictions = self(features[:, :2])

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)
    
    
class RankingModel_(tf.keras.Model):

    def __init__(self, book_title_stringlookup, user_id_stringlookup, genre_lookup):
        super().__init__()
        embedding_dimension = 32
        genre_emb_dim = 4

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([user_id_stringlookup,
          tf.keras.layers.Embedding(user_id_stringlookup.vocabulary_size()+1, embedding_dimension)
        ])

        # Compute embeddings for books.
        self.book_embeddings = tf.keras.Sequential([book_title_stringlookup,
          tf.keras.layers.Embedding(book_title_stringlookup.vocabulary_size()+1, embedding_dimension)
        ])
        
        self.user_genre_1_embeddings = tf.keras.Sequential([genre_lookup,
          tf.keras.layers.Embedding(genre_lookup.vocabulary_size()+1, genre_emb_dim)
        ])
        self.user_genre_2_embeddings = tf.keras.Sequential([genre_lookup,
          tf.keras.layers.Embedding(genre_lookup.vocabulary_size()+1, genre_emb_dim)
        ])
        self.user_genre_3_embeddings = tf.keras.Sequential([genre_lookup,
          tf.keras.layers.Embedding(genre_lookup.vocabulary_size()+1, genre_emb_dim)
        ])
        self.user_genre_4_embeddings = tf.keras.Sequential([genre_lookup,
          tf.keras.layers.Embedding(genre_lookup.vocabulary_size()+1, genre_emb_dim)
        ])
        self.book_genre_1_embeddings = tf.keras.Sequential([genre_lookup,
          tf.keras.layers.Embedding(genre_lookup.vocabulary_size()+1, genre_emb_dim)
        ])
        self.book_genre_2_embeddings = tf.keras.Sequential([genre_lookup,
          tf.keras.layers.Embedding(genre_lookup.vocabulary_size()+1, genre_emb_dim)
        ])
        self.book_genre_3_embeddings = tf.keras.Sequential([genre_lookup,
          tf.keras.layers.Embedding(genre_lookup.vocabulary_size()+1, genre_emb_dim)
        ])
        self.book_genre_4_embeddings = tf.keras.Sequential([genre_lookup,
          tf.keras.layers.Embedding(genre_lookup.vocabulary_size()+1, genre_emb_dim)
        ])
        
        
        # Compute predictions.
        self.ratings = tf.keras.Sequential([
          tf.keras.layers.Dense(32, activation="relu"),
          tf.keras.layers.Dense(16, activation="relu"),
          tf.keras.layers.Dense(8, activation="relu"),
          tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):

        user_id, book_title, user_genre_cat_1, user_genre_cat_2, user_genre_cat_3, user_genre_cat_4, \
        book_genre_cat_1, book_genre_cat_2, book_genre_cat_3, book_genre_cat_4 = inputs

        user_embedding = self.user_embeddings(user_id)
        book_embedding = self.book_embeddings(book_title)
        user_genre_1_embeddings = self.user_genre_1_embeddings(user_genre_cat_1)
        user_genre_2_embeddings = self.user_genre_2_embeddings(user_genre_cat_2)
        user_genre_3_embeddings = self.user_genre_3_embeddings(user_genre_cat_3)
        user_genre_4_embeddings = self.user_genre_4_embeddings(user_genre_cat_4)
        book_genre_1_embeddings = self.book_genre_1_embeddings(book_genre_cat_1)
        book_genre_2_embeddings = self.book_genre_2_embeddings(book_genre_cat_2)
        book_genre_3_embeddings = self.book_genre_3_embeddings(book_genre_cat_3)
        book_genre_4_embeddings = self.book_genre_4_embeddings(book_genre_cat_4)

        return self.ratings(tf.concat([user_embedding, book_embedding, user_genre_1_embeddings, 
                                       user_genre_2_embeddings, user_genre_3_embeddings, user_genre_4_embeddings,
                                      book_genre_1_embeddings, book_genre_2_embeddings, 
                                      book_genre_3_embeddings, book_genre_4_embeddings], axis=1))
    
class BookRecommenderModel_Rich(tfrs.models.Model):

    def __init__(self, 
                 book_title_stringlookup, 
                 user_id_stringlookup, 
                 genre_lookup):
        
        super().__init__()
        
        self.ranking_model: tf.keras.Model = RankingModel_(book_title_stringlookup, 
                                                          user_id_stringlookup, 
                                                          genre_lookup)
        
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
          loss = tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features) -> tf.Tensor:       
        temp = tuple()
        for i in range(0,10):        
            temp += (features[:, i],)
        return self.ranking_model(temp)

    def compute_loss(self, features, training=False) -> tf.Tensor:

        labels =  tf.strings.to_number(features[:,0])
        rating_predictions = self(features[:, 1:])

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)
    
class DCN(tfrs.Model):

    def __init__(self, use_cross_layer, deep_layer_sizes, projection_dim, str_features, vocabularies):
        super().__init__()

        self.embedding_dimension = 32
        int_features = []
        if not isinstance(str_features, list):
            str_features = list(str_features)
        self._all_features = str_features + int_features
        self._embeddings = {}

#     Compute embeddings for string features.
        for feature_name in str_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
              [tf.keras.layers.StringLookup(
                  vocabulary=vocabulary, mask_token=None),
               tf.keras.layers.Embedding(len(vocabulary) + 1,
                                         self.embedding_dimension)
            ])

        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
              projection_dim=projection_dim,
              kernel_initializer=tf.keras.initializers.HeNormal(seed=None))
        else:
            self._cross_layer = None

        self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
          for layer_size in deep_layer_sizes]

        self._logit_layer = tf.keras.layers.Dense(1)

        self.task = tfrs.tasks.Ranking(
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")]
        )

    def call(self, features):
        # Concatenate embeddings
        embeddings = []
        for feature_name in self._all_features:
            embedding_fn = self._embeddings[feature_name]
            embeddings.append(embedding_fn(features[feature_name]))
            
        try:
            x = tf.concat(embeddings, axis=1)
        except:
            embeddings = [tf.reshape(i, (1,-1)) for i in embeddings]
            x = tf.concat(embeddings, axis=1)

        # Build Cross Network
        if self._cross_layer is not None:
            x = self._cross_layer(x)

        # Build Deep Network
        for deep_layer in self._deep_layers:
            x = deep_layer(x)

        return self._logit_layer(x)

    def compute_loss(self, features, training=False):
        labels = tf.strings.to_number(features.pop("RatingOf5"))
        scores = self(features)
        return self.task(
            labels=labels,
            predictions=scores,
        )