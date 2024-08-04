from telegram.ext import Application, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
import os
import shutil
import torch
from PIL import Image
import cv2
import numpy as np

# возьмем переменные окружения из .env
load_dotenv()

# загружаем токен бота
TOKEN =  os.environ.get("TOKEN") # ВАЖНО !!!!!

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')



# функция команды /start
async def start(update, context):
    await update.message.reply_text('Пришлите фото для распознавания объектов')

# функция для работы с текстом
async def help(update, context):
    await update.message.reply_text(update)



# функция обработки изображения
async def detection(update, context):


    my_message = await update.message.reply_text('Мы получили от тебя фотографию. Идет распознавание объектов...')
    # получение файла из сообщения
    new_file = await update.message.photo[-1].get_file()


    # имя файла на сервере
    os.makedirs('images', exist_ok=True)
    image_name = str(new_file['file_path']).split("/")[-1]
    image_path = os.path.join('images', image_name)
    # скачиваем файл с сервера Telegram в папку images
    await new_file.download_to_drive(image_path)

    # Путь к входному изображению
    input_image_path = image_path

    # Путь для сохранения выходного изображения
    output_image_path = 'IMG_test/город3.jpg'

    # Загрузка изображения
    image = Image.open(input_image_path)

    # Выполнение обнаружения
    results = model(image)

    # Преобразование изображения в формат OpenCV
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Перебор обнаруженных объектов
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        if model.names[int(cls)] == 'person' and conf > 0.5:  # Проверка, что объект - человек и уверенность > 50%
            # Рисование прямоугольника вокруг человека
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Сохранение результата
    cv2.imwrite(output_image_path, img)

    print(f"Изображение с обнаруженными людьми сохранено в {output_image_path}")


    # удаляем предыдущее сообщение от бота
    await context.bot.deleteMessage(message_id = my_message.message_id, # если не указать message_id, то удаляется последнее сообщение
                                    chat_id = update.message.chat_id) # если не указать chat_id, то удаляется последнее сообщение

    # отправляем пользователю результат
    await update.message.reply_text('Распознавание объектов завершено') # отправляем пользователю результат 
    await update.message.reply_photo(output_image_path) # отправляем пользователю результат изображение



def main():

    # точка входа в приложение
    application = Application.builder().token(TOKEN).build() # создаем объект класса Application
    print('Бот запущен...')

    # добавляем обработчик команды /start
    application.add_handler(CommandHandler("start", start))
    # добавляем обработчик изображений, которые загружаются в Telegram в СЖАТОМ формате
    # (выбирается при попытке прикрепления изображения к сообщению)
    application.add_handler(MessageHandler(filters.PHOTO, detection, block=False))
    application.add_handler(MessageHandler(filters.TEXT, help))

    application.run_polling() # запускаем бота (остановка CTRL + C)


if __name__ == "__main__":
    main()
