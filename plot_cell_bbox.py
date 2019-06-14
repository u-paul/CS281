from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

record_df = pd.read_csv('../data/p_data/model/record.csv')
r_epochs  = len(record_df)

# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')
# plt.title('mean_overlapping_bboxes')
# plt.subplot(1,2,2)
# plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
# plt.title('class_acc')
# plt.show()

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
plt.title('RPN Classification Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1,2,2)
plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
plt.title('RPN Regression Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
plt.title('Object Classification Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1,2,2)
plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
plt.title('Object Regression Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()