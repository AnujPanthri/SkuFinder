# SKU Finder

## Tasks 

- [ ] **Frame Selection (Part 1)**
    - [x] get_frames : function to get frames from video
    - [ ] filter_frames : function to filter out bad/blurry frames

- [ ] **Product Detection (Part 2)**
    - [ ] Training 
        - [ ] download/gather appropriate dataset
        - [ ] visualize the dataset
        - [ ] choose a model
        - [ ] training the model
        - [ ] evaluate the model
        - [ ] save the model

    - [X] Inference
        - [x] model creation code 
        - [x] model weights loading code
        - [x] model's output to xmin,ymin,xmax,ymax bounding box coordinates for each detection
        - [X] bounding box to crops array
    
    - [ ] Evaluation
        - [ ] load the model using the functions in inference folder
        - [ ] evaluate the model's performance on some dataset and metric
    
- [ ] **Crop Selection (Part 3)**
    - [ ] filter_crops : function to filter out bad/unusable/blurry/very low res crops
    - [x] very low res crops
    - [ ] blurry crops

- [ ] **Product Recognition (Part 4)**
    - [ ] Training
        - [ ] download rp2k dataset
        - [ ] loading and clean the dataset
        - [ ] visualize the dataset
        - [ ] choose a model
        - [ ] train the model
        - [ ] evaluate the model (on some appropriate dataset and metric)
        - [ ] save the model (only the backbone feature extractor)
    
    - Inference
        - [ ] model creation code
        - [ ] model weights loading code
        - [ ] get_embedding : function for getting feature vectors
        - [ ] get_simmilarity_scores : function which takes two feature vectors and compare then and gives the simmilarity scores 0 to 1
        - [ ] get_knn : returns sklearn K nearest neighbours model
    
    - [ ] Evaluation
        - [ ] load the model using the functions in inference folder
        - [ ] evaluate the model's performance on some dataset and metric

- [ ] **Pipeline (Part 5)**
    - [ ] create_database : function to create database of products
    - [ ] load_database : function to load products database
    - [ ] image_to_products : find all product skus in a image and return crops with their guessed sku id
    - [ ] video_to_products : find all product skus in a video and return crops with their guessed sku id
