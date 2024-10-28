# main.py
#from empqt import email_manager,prompt_model,prompt_service_email,config

from logger_config import logger

from email_manager import EmailManager

from prompt_model import Prompt
from prompt_service_email import input_prompt, get_one_to_send, update_doc
import datetime

from time import sleep

import os
#import logging
import sys


def main():
    e_manager = EmailManager()

    ######### main loop #######
    while True:
        # se buscan emails no leidos
        unread_emails = e_manager.get_unread_emails()

        # loop: guardado emails no leidos en db
        for email_dict in unread_emails:
            prompt = email_dict['body']
            #date = datetime(2024, 9, 22, 13, 45, 30)
            email = email_dict['from']
            prompt_to_input = Prompt(input=prompt, email=email)
            input_prompt(prompt_to_input)

        
        # loop: buscando desde la db para enviar
        is_a_doc_to_send = True
        while is_a_doc_to_send:
            doc_to_send = get_one_to_send()
            if doc_to_send is not None:
                document_id = str(doc_to_send['_id'])
                destinatarios = [doc_to_send['email']]
                asunto = "Respuesta automatica a su consulta"
                consulta = doc_to_send['input']
                mensaje = doc_to_send['output']
                if e_manager.send_email(asunto, consulta, mensaje, destinatarios):
                    # marcarlo como enviado
                    update_doc(document_id, "sent")
                else:
                    update_doc(document_id, "error_sending")                    
            else:
                is_a_doc_to_send = False

        sleep(int(e_manager.email_rest_sec))

    ############### testeo para enviar emails #####################


    # # Definir los parámetros para el envío del correo
    # destinatarios = ["charlyvare19@gmail.com", "fedepacher@gmail.com"] #["charlyvare19@gmail.com"] 
    # asunto = "Prueba email con logo 3"
    # mensaje = "Este es un mensaje de prueba..."

    # # Enviar el correo electrónico usando el método send_email
    # e_manager.send_email(asunto, mensaje, destinatarios)


    # ########## testeo para obtener emails no leidos ##############
    # # unread_emails = e_manager.get_unread_emails()
    # # print(unread_emails)
    # # de esta anterior se obtiene algo como esto:
    # unread_emails = [{'from': 'Carlos M Varela <charlyv@fceia.unr.edu.ar>', 'date': datetime.datetime(2024, 9, 14, 19, 21, 1, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=75600))), 'subject': 'Re: Test email desde la clase EmailManager - 01', 'body': 'Hola 12345'}, {'from': 'char var <charlyvare19@gmail.com>', 'date': datetime.datetime(2024, 9, 28, 12, 16, 46, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=75600))), 'subject': 'Re: Test email from Python', 'body': 'Respuesta desde gmail\r\n\r\nEl sáb, 7 sept 2024 a las 18:51, <tambo2022@zohomail.com> escribió:\r\n\r\n> Si esto funciona, ya está lo basico al menos para probar con emails, hasta\r\n> que consigamos alguna cuanta. Por el momento con esta se podria, al menos\r\n> para probar\r\n'}]


    # # ########## testeo para guardar en la mongo ###############
    # # prompt = "Que es un FET?"
    # # #date = datetime(2024, 9, 22, 13, 45, 30)
    # # email = "charlyv@fceia.unr.edu.ar"
    # # prompt_to_input = Prompt(input=prompt, email=email)
    # # input_prompt(prompt_to_input)

    # prompt = unread_emails[0]['body']
    # #date = datetime(2024, 9, 22, 13, 45, 30)
    # email = unread_emails[0]['from']
    # prompt_to_input = Prompt(input=prompt, email=email)
    # input_prompt(prompt_to_input)





    # ########## testeo para obtener de la mongo ###############
    # doc_to_send = get_one_to_send()
    # if doc_to_send is not None:
    #     document_id = str(doc_to_send['_id'])
    #     print(f"type(doc_to_send): {type(doc_to_send)}")
    #     print(f"doc_to_send: {doc_to_send}")
    #     print(f"document_id: {document_id}")

    #     print("enviando email...")
    #     destinatarios = [doc_to_send['email']]
    #     asunto = "Respuesta automatica a su consulta"
    #     mensaje = doc_to_send['output']
    #     e_manager.send_email(asunto, mensaje, destinatarios)

    #     ########## testeo para update la mongo ##############
    #     update_doc(document_id, "sent")



if __name__ == "__main__":
    main()
