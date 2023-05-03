#coding: utf-8
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import time
 
receiver = '986827989@qq.com'
sender = 'powang361@163.com'
smtpserver = 'smtp.163.com'
username = 'powang361@163.com'
password = 'MHNFOBBYVCDYSSDY'
 
mail_title = '训练错误提醒'
mail_body = '训练时发生错误！' + \
	'\n时间： '+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n \
	错误信息如下：\n'

def send_email_tome(mail_title,mail_body):

	message = MIMEText( mail_body, 'plain', 'utf-8' )
	message ['From'] = sender
	message['To'] = receiver
	message['Subject'] = Header( mail_title, 'utf-8' )

	smtp = smtplib.SMTP()                                                     #创建一个连接
	smtp.connect( smtpserver )                                            #连接发送邮件的服务器
	smtp.login( username, password )                                #登录服务器
	smtp.sendmail( sender, receiver, message.as_string() )      #填入邮件的相关信息并发送
	smtp.quit()

def send_error_notification(error):
	send_email_tome(mail_title,mail_body + error)

if __name__ == '__main__':
	send_email_tome(mail_title,mail_body)