#!/bin/bash
# Run this from the top-level!
s=$(grep projectname .env)
s=${s#*'"'}; s=${s%'"'*}
echo "Project name is: $s"

# Fix the setup.py names
sed -i '' "s/projectname/$s/g" setup.py

