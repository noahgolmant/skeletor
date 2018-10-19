#!/bin/bash
# Run this from the top-level!
s=$(grep projectname .env)
s=${s#*'"'}; s=${s%'"'*}
echo "Project name is: $s"

case "$OSTYPE" in
  linux*)   IS_MAC="false" ;;
  darwin*)  IS_MAC="true" ;; 
  *)        echo "os type $OSTYPE unsupported" ;;
esac

if [ "$IS_MAC" = true ] ; then
    sed -i '' "s/projectname/$s/g" setup.py
else
    sed -i "s/projectname/$s/g" setup.py
fi

