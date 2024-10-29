import random
import string


def generate_password():
    length = random.randint(12, 32)
    allowed_punctuation = "#$%&()*+-/<=>?[]{}"
    characters = string.ascii_letters + string.digits + allowed_punctuation
    password = ''.join(random.choice(characters) for i in range(length))
    return password
