#!/bin/sh
memcached -u nobody -m 64 -p 11211 -l 127.0.0.1 &

visdom -port 3000 &
echo "Visdom started on port 3000"

# Execute run.py or similar application if needed
# python3 run.py sst2

tail -f /dev/null
