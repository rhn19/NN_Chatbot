import os
import re
import csv
import codecs

BASE_FOLDER = os.getcwd()
data_folder = r"cornell movie-dialogs corpus"

#split each line into a dict of fields
def loadLines(fileName, fields):
    """
    Split each line into a dictionary of fields
    @param fileName (str) : Path to the file containing line numbers & corresponding text
    @param fields (List[str]) : List of fields to split upon
    @returns lines (Dict{Dict{str:str}}) : Dictionary of fields
    """
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines

def loadConversations(fileName, lines, fields):
    """
    @param fileName (str) : Path to the file containing conversations
    @param lines (Dict{Dict{str:str}}) : Dictionary of fields
    @param fields (List[str]) : List of fields to split upon
    @returns conversations (List[Dict{str}]) : List of conversations as dict
    """
    conversations = []
    with open(fileName, "r", encoding="iso-8859-1") as f:
        for line in f:
            values = line.split(" +++$+++ ")
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            #convert text to list
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

def extractSentencePairs(conversations):
    """
    @param conversations (List[Dict{str}]) : List of conversations as list of line numbers
    @returns sentPairs (List[List[str]]) : List of question answer pairs
    """
    sentPairs = []
    for conv in conversations:
        for i in range(len(conv["lines"])-1):
            inputLine = conv["lines"][i]["text"].strip()
            targetLine = conv["lines"][i+1]["text"].strip()
            #some text fields are empty
            if inputLine and targetLine:
                sentPairs.append([inputLine, targetLine])
    return sentPairs

def savePairs(base_path, output_file, line_fields, convo_fields, delimiter='\t'):
    """
    Writes Formatted Pairs to a file
    @param base_path (str) : Path to the base directory (data)
    @param output_file (str) : Path to the formatted file to write
    @param line_fields (List) : Items in lines file to populate dictionary
    @param convo_fields (List) : Items in conversations file to populate dictionary
    @param delimiter (str) : Delimiter for separating sentence pairs
    """
    line_path = os.path.join(base_path, r"movie_lines.txt")
    convo_path = os.path.join(base_path, r"movie_conversations.txt")

    print("Loading Corpus...")
    lines = loadLines(line_path, line_fields)
    print("Loading Conversations...")
    convos = loadConversations(convo_path, lines, convo_fields)
    print("Extracting Sentence Pairs...")
    qaPairs = extractSentencePairs(convos)

    print("Writing pairs to file...")
    with open(output_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter, lineterminator='\n')
        for pair in qaPairs:
            writer.writerow(pair)

if __name__ == "__main__":
    path = os.path.join(BASE_FOLDER, "data", data_folder)
    outFile = os.path.join(BASE_FOLDER, "generated", r"formatted_pairs.txt")

    if not os.path.exists(path):
        print("Data Directory does not exist!")
        exit(1)
    if not os.path.exists(os.path.join(BASE_FOLDER, "generated")):
        os.mkdir(os.path.join(BASE_FOLDER, "generated"))

    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    savePairs(path, outFile, MOVIE_LINES_FIELDS, MOVIE_CONVERSATIONS_FIELDS, delimiter=delimiter)