# How to do a Self-hosted deployment of LangGraph

Table of contents

- [How it works](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#how-it-works)
- [Helm Chart](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#helm-chart)
- [Environment Variables](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#environment-variables)
- [Build the Docker Image](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#build-the-docker-image)
- [Running the application locally](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#running-the-application-locally)

  - [Using Docker](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#using-docker)
  - [Using Docker Compose](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#using-docker-compose)

1.  [Home](https://langchain-ai.github.io/langgraph/)
2.  [Guides](https://langchain-ai.github.io/langgraph/how-tos/)
3.  [How-to Guides](https://langchain-ai.github.io/langgraph/how-tos/)
4.  [LangGraph Platform](https://langchain-ai.github.io/langgraph/how-tos#langgraph-platform)
5.  [Deployment](https://langchain-ai.github.io/langgraph/how-tos#deployment)

# How to do a Self-hosted deployment of LangGraph[¶](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#how-to-do-a-self-hosted-deployment-of-langgraph "Permanent link")

Prerequisites

- [Application Structure](./Application_Structure_.md)
- [Deployment Options](./Deployment_Options_.md)

This how-to guide will walk you through how to create a docker image from an existing LangGraph application, so you can deploy it on your own infrastructure.

## How it works[¶](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#how-it-works "Permanent link")

With the self-hosted deployment option, you are responsible for managing the infrastructure, including setting up and maintaining necessary databases, Redis instances, and other services.

You will need to do the following:

1.  Deploy Redis and Postgres instances on your own infrastructure.
2.  Build a docker image with the [LangGraph Server](https://langchain-ai.github.io/langgraph/concepts/langgraph_server/) using the [LangGraph CLI](https://langchain-ai.github.io/langgraph/concepts/langgraph_cli/).
3.  Deploy a web server that will run the docker image and pass in the necessary environment variables.

## Helm Chart[¶](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#helm-chart "Permanent link")

If you would like to deploy LangGraph Cloud on Kubernetes, you can use this [Helm chart](https://github.com/langchain-ai/helm/blob/main/charts/langgraph-cloud/README.md).

## Environment Variables[¶](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#environment-variables "Permanent link")

You will eventually need to pass in the following environment variables to the LangGraph Deploy server:

- `REDIS_URI`: Connection details to a Redis instance. Redis will be used as a pub-sub broker to enable streaming real time output from background runs. The value of `REDIS_URI` must be a valid [Redis connection URI](https://redis-py.readthedocs.io/en/stable/connections.html#redis.Redis.from_url).

  Shared Redis Instance

  Multiple self-hosted deployments can share the same Redis instance. For example, for `Deployment A`, `REDIS_URI` can be set to `redis://<hostname_1>:<port>/1` and for `Deployment B`, `REDIS_URI` can be set to `redis://<hostname_1>:<port>/2`.

  `1` and `2` are different database numbers within the same instance, but `<hostname_1>` is shared. **The same database number cannot be used for separate deployments**.

- `DATABASE_URI`: Postgres connection details. Postgres will be used to store assistants, threads, runs, persist thread state and long term memory, and to manage the state of the background task queue with 'exactly once' semantics. The value of `DATABASE_URI` must be a valid [Postgres connection URI](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING-URIS).

  Shared Postgres Instance

  Multiple self-hosted deployments can share the same Postgres instance. For example, for `Deployment A`, `DATABASE_URI` can be set to `postgres://<user>:<password>@/<database_name_1>?host=<hostname_1>` and for `Deployment B`, `DATABASE_URI` can be set to `postgres://<user>:<password>@/<database_name_2>?host=<hostname_1>`.

  `<database_name_1>` and `database_name_2` are different databases within the same instance, but `<hostname_1>` is shared. **The same database cannot be used for separate deployments**.

- `LANGSMITH_API_KEY`: (If using [Self-Hosted Lite](https://langchain-ai.github.io/langgraph/concepts/deployment_options/#self-hosted-lite)) LangSmith API key. This will be used to authenticate ONCE at server start up.
- `LANGGRAPH_CLOUD_LICENSE_KEY`: (If using [Self-Hosted Enterprise](https://langchain-ai.github.io/langgraph/concepts/deployment_options/#self-hosted-enterprise)) LangGraph Platform license key. This will be used to authenticate ONCE at server start up.
- `LANGCHAIN_ENDPOINT`: To send traces to a [self-hosted LangSmith](https://docs.smith.langchain.com/self_hosting) instance, set `LANGCHAIN_ENDPOINT` to the hostname of the self-hosted LangSmith instance.

## Build the Docker Image[¶](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#build-the-docker-image "Permanent link")

Please read the [Application Structure](https://langchain-ai.github.io/langgraph/concepts/application_structure/) guide to understand how to structure your LangGraph application.

If the application is structured correctly, you can build a docker image with the LangGraph Deploy server.

To build the docker image, you first need to install the CLI:

`[](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-0-1)pip install -U langgraph-cli`

You can then use:

`[](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-1-1)langgraph build -t my-image`

This will build a docker image with the LangGraph Deploy server. The `-t my-image` is used to tag the image with a name.

When running this server, you need to pass three environment variables:

## Running the application locally[¶](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#running-the-application-locally "Permanent link")

### Using Docker[¶](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#using-docker "Permanent link")

`[](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-2-1)docker run \     [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-2-2)    --env-file .env \    [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-2-3)    -p 8123:8000 \    [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-2-4)    -e REDIS_URI="foo" \    [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-2-5)    -e DATABASE_URI="bar" \    [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-2-6)    -e LANGSMITH_API_KEY="baz" \    [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-2-7)    my-image`

If you want to run this quickly without setting up a separate Redis and Postgres instance, you can use this docker compose file.

Note

- You need to replace `my-image` with the name of the image you built in the previous step (from `langgraph build`). and you should provide appropriate values for `REDIS_URI`, `DATABASE_URI`, and `LANGSMITH_API_KEY`.
- If your application requires additional environment variables, you can pass them in a similar way.
- If using [Self-Hosted Enterprise](https://langchain-ai.github.io/langgraph/concepts/deployment_options/#self-hosted-enterprise), you must provide `LANGGRAPH_CLOUD_LICENSE_KEY` as an additional environment variable.

### Using Docker Compose[¶](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#using-docker-compose "Permanent link")

`[](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-1)volumes:     [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-2)    langgraph-data:        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-3)        driver: local [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-4)services:     [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-5)    langgraph-redis:        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-6)        image: redis:6        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-7)        healthcheck:            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-8)            test: redis-cli ping            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-9)            interval: 5s            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-10)            timeout: 1s            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-11)            retries: 5    [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-12)    langgraph-postgres:        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-13)        image: postgres:16        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-14)        ports:            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-15)            - "5433:5432"        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-16)        environment:            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-17)            POSTGRES_DB: postgres            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-18)            POSTGRES_USER: postgres            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-19)            POSTGRES_PASSWORD: postgres        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-20)        volumes:            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-21)            - langgraph-data:/var/lib/postgresql/data        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-22)        healthcheck:            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-23)            test: pg_isready -U postgres            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-24)            start_period: 10s            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-25)            timeout: 1s            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-26)            retries: 5            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-27)            interval: 5s    [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-28)    langgraph-api:        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-29)        image: ${IMAGE_NAME}        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-30)        ports:            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-31)            - "8123:8000"        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-32)        depends_on:            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-33)            langgraph-redis:                [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-34)                condition: service_healthy            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-35)            langgraph-postgres:                [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-36)                condition: service_healthy        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-37)        env_file:            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-38)            - .env        [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-39)        environment:            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-40)            REDIS_URI: redis://langgraph-redis:6379            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-41)            LANGSMITH_API_KEY: ${LANGSMITH_API_KEY}            [](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-3-42)            POSTGRES_URI: postgres://postgres:postgres@langgraph-postgres:5432/postgres?sslmode=disable`

You can then run `docker compose up` with this Docker compose file in the same folder.

This will spin up LangGraph Deploy on port `8123` (if you want to change this, you can change this by changing the ports in the `langgraph-api` volume).

You can test that the application is up by checking:

`[](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-4-1)curl --request GET --url 0.0.0.0:8123/ok`

Assuming everything is running correctly, you should see a response like:

`[](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#__codelineno-5-1){"ok":true}`
