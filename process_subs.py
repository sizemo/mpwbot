# python
import os

sub_path = "/Users/alexandra/Documents/Programming/mpwbot/subtitles"

def extract_transcripts_from_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        transcript = []
        for line in lines:
            if '-->' not in line and line.strip():
                transcript.append(line.strip())
        return '\n'.join(transcript)

def process_transcripts(directory):
    all_transcripts = []
    for filename in os.listdir(directory):
        if filename.endswith('.srt'):
            file_path = os.path.join(directory, filename)
            transcript = extract_transcripts_from_srt(file_path)
            all_transcripts.append(transcript)
    return '\n'.join(all_transcripts)

transcripts = process_transcripts(sub_path)