# Documentation

## Dependencies

```shell
#poetry install  # setup dependencies
poetry self update
poetry install --no-root # without install the current project
poetry add package  # add new python packages
```

## Start service

```shell
# run script
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# uvicorn main:app --host 0.0.0.0 --log-config log_conf.yaml
uvicorn main:app --log-config log_conf.yaml
#uvicorn main:app --reload

```

## database

### db migration

We use [aerich](https://github.com/tortoise/aerich) for db migration. Cheatsheet:

- `aerich init -t your.app.path.to.TORTOISE_ORM` --> initiate aerich step 1
- `aerich init-db` --> initiate aerich step 2
- `aerich migrate --name drop_column` --> run after db change to commit
- `aerich upgrade` --> run to update db
- `aerich downgrade` --> run to roll back
- `aerich history` --> show commit history
- `aerich heads` --> show commit head

For more usages use `-h` or checkout the project on GitHub!

### generate schema

We use MySQL and Pinecone for state persistence.
To get schema for MySQL:

```python
from app.utils import get_schema

print(get_schema())
```

Then import generated sql code to [dbdiagram.io](https://dbdiagram.io/home)
for ER diagram visualization.

## docker image

Commands for building images:

```shell
# build docker image
docker build -t sentia-mapreduce .

# run container
docker run -p 8080:8000 --rm --name your-service sentia-mapreduce

# remove dangling images
docker rmi $(docker images -f "dangling=true" -q)

# or a more ci way is to run docker compose in the `sentia-gateway` project  
docker-compose -f docker-compose.yml up --build
```

## testing

Add test script in the `tests/` folder for testing, and run with:

```shell
# This tests `test_ingest` in `test_ingestion.py`
pytest tests/test_ingestion.py::test_ingest --verbose

```

Checkout `pytest.ini` for controlling system path, log level etc.

## deployment

Two things to remember:

1. remember to update the `.env` file in the project in the test environment since git does not.
2. remember to update the `deploy` folder since k8s does not.

## kafka

For dev environment, install kafka and run it locally:

```shell
zookeeper-server-start /usr/local/etc/kafka/zookeeper.properties
kafka-server-start /usr/local/etc/kafka/server.properties
```

On macOS, using homebrew to start/stop zookeeper and kafka background service
(will persist through reboot).

```shell
# install
brew install kafka
# start
brew services start zookeeper
brew services start kafka
# stop
brew services stop kafka
brew services stop zookeeper

```

### GUI monitor [kafka-ui](https://github.com/provectus/kafka-ui)

With **kafka-ui** we can view traffics on topics and messages in real time:

- ready to use image, just pull from dockerhub
- view in browser, e.g. http://192.168.9.32:8080
- send messages to topic

```shell
# detach mode
docker run -d --rm -p 8080:8080 -e DYNAMIC_CONFIG_ENABLED=true -e KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS=192.168.9.32:9092 provectuslabs/kafka-ui
# or interactively
docker run --rm -it -p 8080:8080 -e DYNAMIC_CONFIG_ENABLED=true -e KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS=192.168.9.32:9092 provectuslabs/kafka-ui

```

Or to do it in the terminal:

```shell
kafka-topics --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic test
kafka-console-producer --broker-list localhost:9092 --topic test
kafka-console-consumer --bootstrap-server localhost:9092 --topic test --from-beginning

```

### references:

- [Apache Kafka Installation on Mac using Homebrew](https://medium.com/@Ankitthakur/apache-kafka-installation-on-mac-using-homebrew-a367cdefd273)

## Dones and Todos:

- [x] logging remake with loguru
- [ ] **concurrent** openai embedding
- [ ] **concurrent** pinecone upsert
- [x] rerank operational (proxy needed still)
- [x] pydantic upgraded to v2
- [x] pagination with fastapi-pagination
- [x] async iterate embedding and upsert in ingest workflow to reduce memory footprint
- [x] openai sdk upgrade
- [ ] ~~logging remake with loguru~~
- [ ] ~~**concurrent** openai embedding~~
- [ ] ~~**concurrent** pinecone upsert~~
- [x] **vdb abstraction layer** operational
    - **test** scripts done
    - **Pinecone** operational
    - **Tencent** operational
    - **ByteDance** operational
    - **HuaWei** operational
    - general **migration** script operational, pinecone -> others
- [x] add **token limit** to chunk append and chunk update through schema validation
- [x] **start up** optimization with text splitter preload and nltk packages preload
- [x] **QA extraction** operational
- [x] chunks vdb enable/disable operational
- [x] extend chunks CRUD to **QA pairs**
- [x] chunks **CRUD**
- [x] pinecone migrated to **v3**
- [x] api for **deleting file**
- [x] api for **chunking file** by providing url
- [x] Add **QA loader** (.csv) in `file_parse.py`
- [x] **centralize .env load** with module and pydantic
- [x] **pinecone migration** script now available in `script/cmd.py`
- [x] **kafka consumer** operational in `ingest`
- [x] fix **memory leak** from loader class in `file_parse.py`.
- [x] use **queue for schedule** to prevent run out of SQL connections
- [x] fix **deadlock** from using atomic updater (caused by async calls in critical region)
- [x] fix wps docx debacle (by `catdoc`), works in linux, but **not on macOS**
- [x] remade of **pdf loader** now operational with integration with `ingest`
- [x] add api to query file from query text
- [x] custom **pdf loader** that load both images and words from document
  and when `.load()` automatically upload all images to COS and put
  keys into the `metadata` of relevant chunks.
- [x] config **logging** for uvicorn and root through `log_conf.yaml`
- [x] add **COS**(powered by [tencent](https://console.cloud.tencent.com/cos/)) client wrapper
- [x] send message to **kafka MQ** when ingest finished (only when kafka is available)
- [x] complete remake of **ingest workflow** with test cases
- [x] upgrade **proxy** method for openai and pinecone (towards more official ways of doing things)
- [x] add **aerich** tool for db migration with readme
- [x] add api for getting least-used key (**load balancing**)
- [x] add **azure** in api key pool
- [x] add multiple tests for **TDD**
- [x] **typer based cmd scripts** for custom command (caveat: doesn't work with in memory
  sqlite, works with async functions [reference](https://github.com/tiangolo/typer/issues/88))
- [x] **gptCache** + **api key adapter**
- [x] add **pinecone consistency** check and repair
- [x] add openai/azure custom **key adapter** to run each call with dynamic key configurations
- [x] fix warm up issues -> split_text too slow, made it async
- [x] fix reload -> download failure retry
    - [x] openai embed failure retry -> abstracted embed stage block
    - [x] pinecone upsert failure retry -> abstracted to upsert stage block
- [x] fix chunk id collision problem -> revert to uuid for chunk id generation instead of md5
- [x] fix doc unique bug (doc + metadata unique instead of doc only,
- [x] fix upload bugs -> adjust upload logic
  use table to reroute namespace and other fields
- [x] **multi-media support**
    - [x] deprecated `doc` APIs
    - [x] manual unit tests past
    - [x] add ms doc in addition to docx
    - [x] add excel loader
- [ ] count tokens for embeddings
- [x] kafka integration + finish topic
    - [x] project structure reorg
- refactor:
    - [x] remove core/upload.py
    - [ ] remove chat doc apis
