from jina import Client, Document

if __name__ == "__main__":
    client = Client(port=8008)
    result = client.post('/', Document(uri="./demos/gradio/parrot_plushie_1.jpg"))
    print(result[0].text)