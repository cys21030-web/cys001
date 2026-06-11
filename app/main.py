"""
now based on the information we have, create a flask web site

+ init tof sensor when the server start

+ keep pushing data to web client and visualize raw matrix data / world point cloud (toggle on / off)

+ data collecting mode: user first place a prototype of and select 'Normal', 'Upstairs', or 'Downstairs' (depends on the content of the ToFLabels)
    - by clicking snapshot, a set of raw data is saved in server side with timestamp and place it in 'snapshot/<label>/tof_<timestamp>.dat'
    - user can view the collected data in 'snapshot/<label>/' and view the raw data and world point cloud (toggle on / off)
    - user can select many collected data and export them (with their raw_data, json, and world point cloud, ply) as zip

+ training mode: 
    - user can select many collected data and train a model on the server side, the model will be saved in 'model/<timestamp>.pth'
    - user can view the training history and export the trained model as zip
    
+ inference mode:
    - the system keep recording the ToF data
    - user can select a model and start inference
        - when user click a button, the system use the model to predict the next collected data
        - the data (with raw_data and point cloud) and the prediction result will be shown in the web page until use click the accepted (correct) / reject (incorrect) button
        - correct rate and confusion matrix will be updated and shown in the web page
        - generate a report (pdf) when user click the 'generate report' button, the report will include number of inferencing, correct rate, confusion matrix
    
"""
