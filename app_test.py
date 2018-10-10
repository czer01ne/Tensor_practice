import tensorflow as tf

def main(argv=None):
    print("argv:", argv)
    import sys
    print("tf.app.flags.FLAGS.__dict__:", tf.app.flags.FLAGS.__dict__)
    print("sys.argv:", sys.argv)

if __name__ == '__main__':
    tf.app.run()
