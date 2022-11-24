# Classi🔥 - Image to image classification!

**Status**: In research, beta weights available

Classify any image without having to train a new model.

Beta notice: I used this model for another project and it's good enough, now that its a separated project, I'll try make better models.

# Demos

# Jina (Spawn your own classifier API)

Given 1 or more source images, you can spawn a classifier in seconds, for any object you wish.

- Clone this repo
- `cd` to the 'jina' folder inside this repo
- (Optional) make a virtual environment: `python3 -m venv .venv` and activate: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Run `python ./create_jina_flow.py my_image.jpg > classifire_flow.yml` (for example: `python ./create_jina_flow.py ../test_images/parrot_plushie/20221022_214232.jpg > classifire_flow.yml`)
- Congrats! `classifire_flow.yml` can now be used to spawn your API.
- Make sure to be logged into jcloud. If you aren't logged in, run `jc login` and follow further instructions.
- Call `jc deploy classifire_flow.yml` to deploy your custom API.

## Test!

You should have an endpoint (ending with `.wolf.jina.ai`). This is the endpoint to the classifier API you just created. To test the API out, run the following example code using Python:

```py
from jina import Client, Document

if __name__ == "__main__":
    client = Client(host='grpcs://my_url.wolf.jina.ai', asyncio=True)
    result = client.post('/', Document(uri="../test_images/parrot_plushie/20221022_214232.jpg"))
    print(result[0].text)
```

## Where to go from here?

See [Jina docs](https://docs.jina.ai), a lot can be configured, such as protocol usage (http, grpc, etc) as well as port number and monitoring.