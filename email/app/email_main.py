# main.py

from email_manager import EmailManager

from prompt_model import Prompt
from prompt_service_email import input_prompt, get_one_to_send, update_doc
from datetime import datetime

def main():
    ############### testeo para enviar emails #####################
    # Instanciar la clase EmailSender
    e_manager = EmailManager()

    # Definir los parámetros para el envío del correo
    destinatarios = ["charlyv@fceia.unr.edu.ar", "fedepacher@gmail.com"]
    asunto = "Test email desde la clase EmailManager - 03"
    mensaje = "Este es un mensaje de prueba enviado desde la clase EmailManager usando variables de entorno."

    # Enviar el correo electrónico usando el método send_email
    #e_manager.send_email(asunto, mensaje, destinatarios)


    ########## testeo para obtener emails no leidos ##############
    #unread_emails = e_manager.get_unread_emails()
    #print(unread_emails)


    ########## testeo para guardar en la mongo ###############
    prompt = "Que es un FET?"
    #date = datetime(2024, 9, 22, 13, 45, 30)
    email = "charlyv@fceia.unr.edu.ar"
    prompt_to_input = Prompt(input=prompt, email=email)
    input_prompt(prompt_to_input)


    ########## testeo para obtener de la mongo ###############
    doc_to_send = get_one_to_send()
    if doc_to_send is not None:
        document_id = str(doc_to_send['_id'])
        print(f"type(doc_to_send): {type(doc_to_send)}")
        print(f"doc_to_send: {doc_to_send}")
        print(f"document_id: {document_id}")

        print("enviando email...")
        destinatarios = [doc_to_send['email']]
        asunto = "Respuesta automatica a su consulta"
        mensaje = doc_to_send['output']
        e_manager.send_email(asunto, mensaje, destinatarios)

        ########## testeo para update la mongo ##############
        update_doc(document_id, "sent")



if __name__ == "__main__":
    main()
