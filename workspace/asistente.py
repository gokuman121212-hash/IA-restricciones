import datetime

def responder(comando):
    if comando.lower() == "hola":
        return "¡Hola! ¿Cómo estás?"
    elif comando.lower() == "adios":
        return "¡Adiós! ¡Hasta luego!"
    elif comando.lower() == "que hora es":
        return datetime.datetime.now().strftime("%H:%M:%S")
    else:
        return "Lo siento, no entiendo ese comando."

def main():
    comandos = ["hola", "adios", "que hora es", "otro comando"]
    for comando in comandos:
        respuesta = responder(comando)
        print(f"Comando: {comando}, Respuesta: {respuesta}")

if __name__ == "__main__":
    main()
