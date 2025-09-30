#!/bin/bash

# Set the channel URL
CHANNEL_URL="https://www.youtube.com/@mpwdigital6243/streams/"

# Use yt-dlp with a filter for video titles
yt-dlp -P '/Users/alexandra/Documents/Programming/mpwbot/subtitles' "$CHANNEL_URL" --skip-download\
 --write-auto-subs --sub-format srt --match-filters "title ~= ^(BlueSky Live|Oxford Exxon Podcast)" "was_live = TRUE"\
 --sleep-requests 35