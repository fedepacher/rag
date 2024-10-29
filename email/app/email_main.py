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


if __name__ == "__main__":
    main()
