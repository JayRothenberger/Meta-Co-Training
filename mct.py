import numpy as np
import torch
import torchvision
import os
import time
from utils import VATLoss

vat_loss = VATLoss()

def MCT(U_t, U_s, L_t, L_s, Y_t, Y_s, teacher, student, teacher_optimizer, student_optimizer, loss=torch.nn.CrossEntropyLoss(), supervised=False, approx=False, previous_params=True):
    device = int(os.environ["RANK"]) % torch.cuda.device_count()
    
    SPL_t = teacher(U_t) # compute the soft pseudo labels (teacher)
    PL_t = torch.tensor([np.random.choice(np.arange(SPL_t.shape[-1]), None, False, torch.nn.Softmax(-1)(SPL_t.type(torch.float32)).detach().cpu().numpy()[xi]) for xi in range(SPL_t.shape[0])]).to(device) # sample from the SPL distribution
    # PL_t = torch.argmax(SPL_t, -1)
    if previous_params:
        self_loss_t = loss(SPL_t, PL_t)
        self_loss_t.backward()
        self_loss_t_grads = [param.grad.detach().clone() for param in teacher.parameters()]
        teacher_optimizer.zero_grad()
    
    SPL_s = student(U_s) # compute the soft pseudo labels (student)
    PL_s = torch.tensor([np.random.choice(np.arange(SPL_s.shape[-1]), None, False, torch.nn.Softmax(-1)(SPL_s.type(torch.float32)).detach().cpu().numpy()[xi]) for xi in range(SPL_s.shape[0])]).to(device) # sample from the SPL distribution
    # PL_s = torch.argmax(SPL_s, -1)
    if previous_params:
        self_loss_s = loss(SPL_s, PL_s)
        self_loss_s.backward()
        self_loss_s_grads = [param.grad.detach().clone() for param in student.parameters()]
        student_optimizer.zero_grad()
        
    # compute the gradient of the student parameters with respect to the pseudo labels
    if approx:
        student_initial_output = student(L_s)
        student_loss_initial_l = loss(student_initial_output, Y_s).detach().clone()

        teacher_initial_output = teacher(L_t)
        teacher_loss_initial_l = loss(teacher_initial_output, Y_t).detach().clone()
        
    student_optimizer.zero_grad()
    teacher_optimizer.zero_grad()
    
    student_initial_output = student(U_s)
    student_loss_initial = loss(student_initial_output, PL_t)

    
    student_optimizer.zero_grad()
    student_loss_initial.backward()
    if not approx:
        grads1_s = [param.grad.data.detach().clone() for param in student.parameters()]

    # update the student parameters based on pseudo labels from teacher
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    student_optimizer.step()
    student_optimizer.zero_grad()
    
    teacher_initial_output = teacher(U_t)
    teacher_loss_initial = loss(teacher_initial_output, PL_s)

    teacher_optimizer.zero_grad()
    teacher_loss_initial.backward()
    if not approx:
        grads1_t = [param.grad.data.detach().clone() for param in teacher.parameters()]
    
    # update the teacher parameters based on pseudo labels from student
    torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
    teacher_optimizer.step()
    teacher_optimizer.zero_grad()
    
    # compute the gradient of the student parameters with respect to the real labels
    student_final_output = student(L_s)
    student_loss_final = loss(student_final_output, Y_s)

    if not approx:
        student_loss_final.backward()
        h_t = sum([(param.grad.data.detach() * grads).sum() for param, grads in zip(student.parameters(),  grads1_s)])

    student_optimizer.zero_grad()
        
    # compute the gradient of the student parameters with respect to the real labels
    teacher_final_output = teacher(L_t)
    teacher_loss_final = loss(teacher_final_output, Y_t)

    if not approx:
        teacher_loss_final.backward()
        h_s = sum([(param.grad.data.detach() * grads).sum() for param, grads in zip(teacher.parameters(),  grads1_t)])

    teacher_optimizer.zero_grad()

    # compute the teacher MPL loss
    if not previous_params:
        if approx:
            # https://github.com/google-research/google-research/issues/536
            # h is approximable by: student_loss_final - loss(student_initial(L), Y) where student_initial is before the gradient update for U
            h_approx_t = student_loss_initial_l - student_loss_final
            h_approx_s = teacher_loss_initial_l - teacher_loss_final
            # this is the first order taylor approximation of the above loss, and apparently has finite deviation from the true quantity.
            # for correctness, I include instead the theoretically correct computation of h
            student_loss_mpl = h_approx_s.detach() * loss(SPL_t, PL_t)
            teacher_loss_mpl = h_approx_t.detach() * loss(SPL_s, PL_s)
        else:
            SPL_t = teacher(U_t) # (re) compute the soft pseudo labels (teacher)
            SPL_s = student(U_s) # (re) compute the soft pseudo labels (student)
            
            teacher_loss_mpl = h_t.detach() * loss(SPL_t, PL_t)
            student_loss_mpl = h_s.detach() * loss(SPL_s, PL_s)
    else:
        if approx:
            # https://github.com/google-research/google-research/issues/536
            # h is approximable by: student_loss_final - loss(student_initial(L), Y) where student_initial is before the gradient update for U
            h_t = student_loss_initial_l - student_loss_final
            h_s = teacher_loss_initial_l - teacher_loss_final
            # this is the first order taylor approximation of the above loss, and apparently has finite deviation from the true quantity.
            # for correctness, I include instead the theoretically correct computation of h
        # it is already computed above
        teacher_loss_mpl = 0.0
        student_loss_mpl = 0.0
    
    if supervised:# optionally compute the supervised loss
        student_out = student(L_s)
        student_loss_sup = loss(student_out, Y_s)
        student_loss = student_loss_mpl + student_loss_sup
        
    else:
        student_loss = student_loss_mpl
    # update student based on teacher performance
    student_optimizer.zero_grad()
    
    if student_loss != 0.0:
        student_loss.backward()
    
    if previous_params: # compute the MPL update based on the original parameters
        for param, grad in zip(student.parameters(), self_loss_s_grads):
            if param.grad is not None:
                param.grad += h_s * grad
            else:
                param.grad = h_s * grad
        
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    student_optimizer.step()
    student_optimizer.zero_grad()

    if supervised:# optionally compute the supervised loss
        teacher_out = teacher(L_t)
        teacher_loss_sup = loss(teacher_out, Y_t)
        teacher_loss = teacher_loss_mpl + teacher_loss_sup
    else:
        teacher_loss = teacher_loss_mpl
        
    
    # update teacher based on student performance
    teacher_optimizer.zero_grad()
    if teacher_loss != 0.0:
        teacher_loss.backward()
    
    if previous_params: # compute the MPL update based on the original parameters
        for param, grad in zip(teacher.parameters(), self_loss_t_grads):
            if param.grad is not None:
                param.grad += h_t * grad
            else:
                param.grad = h_t * grad
        
    torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
    teacher_optimizer.step()
    teacher_optimizer.zero_grad()
