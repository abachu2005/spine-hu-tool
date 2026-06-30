"""Cloud segmentation server (the heavy ML step, offloaded).

Runs TotalSegmentator behind a small HTTP API so a low-RAM client (e.g. an
8 GB laptop) never has to run the model locally. Deployable to Cloud Run.
"""
