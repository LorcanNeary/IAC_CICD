import fire

def hello(name="World"):
  return "Hello sad old %s!" % name

if __name__ == '__main__':
  fire.Fire(hello)
