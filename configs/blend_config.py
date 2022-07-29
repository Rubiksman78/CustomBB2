DEFAULT_CONFIG = {
    "USE_CUDA" : False, #Use gpu or not
    "N_TURNS" : 10, #Number of turn for the loop option
    "MODEL":"zoo:blenderbot2/blenderbot2_3B/model", #replace 3B with 400M to try the lighter one
    "PRINT_TERMINAL":True, #Print the response in the terminal
    "RESET_HISTORY":False, #Reset the history file
    "INCLUDE_PERSONA":False, #Include the persona in the context
    "USE_SMALL":True, #Use the small model
    "HISTORY_PATH":"text_files/history.txt", #Path of the history file
    "PERSONA_PATH":"database.csv", #Path of the persona file
    "ID":0} #Id of the persona in the database
