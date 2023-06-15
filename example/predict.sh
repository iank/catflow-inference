#!/bin/sh
curl -X POST "http://localhost:5054/predict/" -H "Content-Type: application/octet-stream" --data-binary "@example.png"
