# EDA findings:
1. Ethnecity / Origin: American movis dominates the move scene followed by britan and indea
2. Movies Released per Year: 
   1. The number of movies were exponentially growing from the 70s
   2. The movies industry was slowing down in the cold-war peroid 
   3. The movies were being heavily produced since the first world war, perhabs because america was not affected by those wars as well as the booming advances in movies and pictures technology
3.  Genres: 
    1.  the data distripution of genres is skewd as the drama and comedy are dominating most of the movie genres, most probably because this is what people around the world love to watch without being strictly to a certain group of age or culture. 
    2.  many of of the genres only occured once in the data which made me decide on only using the top 25 genres as they had a reasnable occurance count
4.  Movie Titles: Many of the movies had sequels or remakes like cindrella and Alice in wonderland, and the most remaked one was cindrella, with many other films sharing the same name, like " love , I love You and Mother " which occured 5 times 
5.  Cast Members: As with the movie titles the sequels had huge impact on the number of mentions of casting members, especially in Cartoons like tom and jerry which appeared in 80 films, and other looney tunes charachters.
6.  Directors: same with Cast members, many of the Cartoon movies had the same directors as with old movies as well, where studios had the same director for multiple movies, the top 10 directors for example directed at least 50 movies and more.
7.  

# Modelling
1. Data Cleaning: by removing duplicates and Unknown genres or missing values  
2. Preprocessing : had two approaches
   1. Using Bert Tokenizer to tokenize the movies plots
   2. Using qwen2.5:3b to summerize the plots first before sending it to the tokenizer, to reduce the number of tokens generated 
3. Model Training: had three approaches - most of them got around 40% accuracy on 25 genres
   1. Using simple machine learning approach on the tokenized plot
   2. Using pretrained tiny_bert to train a classifier based on the produced tokens 
   3. Using pretrained tiny_Bert to train the classifier on the summerized plots 
   
# Summerization without Spoling
1. Used OLLAMA to Run and host qwen2.5:3b model and using the right prompt to make sure no spoiling is found in the summerized data.