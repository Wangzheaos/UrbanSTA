import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from util.metrics import get_MSE

def train_one_epoch(c_model: torch.nn.Module, f_model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    c_model.train(True)
    f_model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        len_num = 0
        if args.len_closeness != 0:
            len_num += 1
        if args.len_period != 0:
            len_num += 1
        if args.len_trend != 0:
            len_num += 1
        if len_num == 1:
            flows_xc, flows_c, flows_d, flows_f = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_c = flows_c.to(device, non_blocking=True)
            flows_d = flows_d.to(device, non_blocking=True)
            flows_f = flows_f.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss1, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, data, _ = c_model(flows_xc, flows_c, flows_d)
                loss2, poif_loss, sf_loss, preds = f_model(data, flows_f, mask_ratio=args.mask_ratio)

        elif len_num == 2:
            flows_xc, flows_xp, flows_c, flows_d, flows_f = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_xp = flows_xp.to(device, non_blocking=True)
            flows_c = flows_c.to(device, non_blocking=True)
            flows_d = flows_d.to(device, non_blocking=True)
            flows_f = flows_f.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss1, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, data, _ = c_model(flows_xc, flows_xp, flows_c, flows_d)
                loss2, poif_loss, sf_loss, preds = f_model(data, flows_f, mask_ratio=args.mask_ratio)
        else:
            flows_xc, flows_xp, flows_xt, flows_c, flows_d, flows_f = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_xp = flows_xp.to(device, non_blocking=True)
            flows_xt = flows_xt.to(device, non_blocking=True)
            flows_c = flows_c.to(device, non_blocking=True)
            flows_d = flows_d.to(device, non_blocking=True)
            flows_f = flows_f.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss1, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, data, _ = c_model(flows_xc, flows_xp, flows_xt, flows_c, flows_d)
                loss2, poif_loss, sf_loss, preds = f_model(data, flows_f, mask_ratio=args.mask_ratio)

        loss = loss1 * args.mu + loss2 + args.alpha * s_loss + args.beta * t_loss \
               + args.d_alpha * sd_loss + args.d_beta * td_loss  + args.delte * sf_loss + args.gama * poi_loss + args.theta * poif_loss


        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=f_model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        loss1_value = math.sqrt(loss1.item())
        loss2_value = math.sqrt(loss2.item())
        s_loss_value = math.sqrt(s_loss.item())
        sd_loss_value = math.sqrt(sd_loss.item())
        t_loss_value = math.sqrt(t_loss.item())
        td_loss_value = math.sqrt(td_loss.item())
        sf_loss_value = math.sqrt(sf_loss.item())

        metric_logger.update(loss=math.sqrt(loss_value))
        metric_logger.update(r_loss=loss1_value)
        metric_logger.update(f_loss=loss2_value)
        metric_logger.update(poi_loss=poi_loss.item())
        metric_logger.update(poif_loss=poif_loss.item())
        metric_logger.update(s_loss=s_loss_value)
        metric_logger.update(sd_loss=sd_loss_value)
        metric_logger.update(t_loss=t_loss_value)
        metric_logger.update(td_loss=td_loss_value)
        metric_logger.update(sf_loss=sf_loss_value)

        clr = optimizer.param_groups[0]["clr"]
        flr = optimizer.param_groups[0]["flr"]

        # metric_logger.update(lr=lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            # epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch)
            log_writer.add_scalar('clr', clr, epoch)
            log_writer.add_scalar('flr', flr, epoch)
            log_writer.add_scalar('r_loss', loss1_value, epoch)
            log_writer.add_scalar('f_loss', loss2_value, epoch)
            log_writer.add_scalar('poi_loss', poi_loss.item(), epoch)
            log_writer.add_scalar('poif_loss', poif_loss.item(), epoch)
            log_writer.add_scalar('s_loss', s_loss_value, epoch)
            log_writer.add_scalar('sd_loss', sd_loss_value, epoch)
            log_writer.add_scalar('t_loss', t_loss_value, epoch)
            log_writer.add_scalar('td_loss', td_loss_value, epoch)
            log_writer.add_scalar('sf_loss', sf_loss_value, epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_all(c_model: torch.nn.Module, f_model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, mmn,
                    log_writer=None,
                    args=None):
    c_model.train(True)
    f_model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        len_num = 0
        if args.len_closeness != 0:
            len_num += 1
        if args.len_period != 0:
            len_num += 1
        if args.len_trend != 0:
            len_num += 1
        if len_num == 1:
            flows_xc, flows_c, flows_d, flows_f = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_c = flows_c.to(device, non_blocking=True)
            flows_d = flows_d.to(device, non_blocking=True)
            flows_f = flows_f.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss1, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, data, latent = c_model(flows_xc, flows_c, flows_d)
                data = data / args.scaler_X
                flows_f = flows_f / args.scaler_Y
                loss2, poif_loss, sf_loss, preds = f_model(data, latent, flows_f, mask_ratio=args.mask_ratio)

        elif len_num == 2:
            flows_xc, flows_xp, flows_c, flows_d, flows_f = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_xp = flows_xp.to(device, non_blocking=True)
            flows_c = flows_c.to(device, non_blocking=True)
            flows_d = flows_d.to(device, non_blocking=True)
            flows_f = flows_f.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss1, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, data, latent = c_model(flows_xc, flows_xp, flows_c, flows_d)
                data = data / args.scaler_X
                flows_f = flows_f / args.scaler_Y
                loss2, poif_loss, sf_loss, preds = f_model(data, latent, flows_f, mask_ratio=args.mask_ratio)
        else:
            flows_xc, flows_xp, flows_xt, flows_c, flows_d, flows_f = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_xp = flows_xp.to(device, non_blocking=True)
            flows_xt = flows_xt.to(device, non_blocking=True)
            flows_c = flows_c.to(device, non_blocking=True)
            flows_d = flows_d.to(device, non_blocking=True)
            flows_f = flows_f.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss1, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, data, latent = c_model(flows_xc, flows_xp, flows_xt, flows_c, flows_d)
                data = data / args.scaler_X
                flows_f = flows_f / args.scaler_Y
                loss2, poif_loss, sf_loss, preds = f_model(data, latent, flows_f, mask_ratio=args.mask_ratio)



        # if epoch >= 200:
        #     t = 0.0001 * math.pow(10, epoch // 100)
        if not math.isfinite(loss2.item()):
            loss2 = torch.where(torch.isinf(loss2), torch.full_like(loss2, 1), loss2)


        loss = loss1 * args.mu + loss2 * args.nu + args.alpha * s_loss + args.beta * t_loss \
               + args.d_alpha * sd_loss + args.d_beta * td_loss  + args.delte * sf_loss + args.gama * poi_loss + args.theta * poif_loss


        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss1.item())
            print(loss2.item())
            print(s_loss.item())
            print(t_loss.item())
            print(poi_loss.item())
            print(poif_loss.item())
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=f_model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        loss1_value = math.sqrt(loss1.item())
        loss2_value = math.sqrt(loss2.item())
        s_loss_value = math.sqrt(s_loss.item())
        sd_loss_value = math.sqrt(sd_loss.item())
        t_loss_value = math.sqrt(t_loss.item())
        td_loss_value = math.sqrt(td_loss.item())
        sf_loss_value = math.sqrt(sf_loss.item())
        poi_loss_value = poi_loss.item()
        poif_loss_value = poif_loss.item()

        metric_logger.update(loss=math.sqrt(loss_value))
        metric_logger.update(r_loss=loss1_value)
        metric_logger.update(f_loss=loss2_value)
        metric_logger.update(poi_loss= poi_loss_value)
        metric_logger.update(poif_loss= poif_loss_value)
        metric_logger.update(s_loss=s_loss_value)
        metric_logger.update(sd_loss=sd_loss_value)
        metric_logger.update(t_loss=t_loss_value)
        metric_logger.update(td_loss=td_loss_value)
        metric_logger.update(sf_loss=sf_loss_value)

        flr = optimizer.param_groups[0]["flr"]
        clr = optimizer.param_groups[0]["clr"]
        # metric_logger.update(lr=lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            # epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch)
            log_writer.add_scalar('clr', clr, epoch)
            log_writer.add_scalar('flr', flr, epoch)
            log_writer.add_scalar('r_loss', loss1_value, epoch)
            log_writer.add_scalar('f_loss', loss2_value, epoch)
            log_writer.add_scalar('poi_loss', poi_loss_value, epoch)
            log_writer.add_scalar('poif_loss', poif_loss_value, epoch)
            log_writer.add_scalar('s_loss', s_loss_value, epoch)
            log_writer.add_scalar('sd_loss', sd_loss_value, epoch)
            log_writer.add_scalar('t_loss', t_loss_value, epoch)
            log_writer.add_scalar('td_loss', td_loss_value, epoch)
            log_writer.add_scalar('sf_loss', sf_loss_value, epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_complete(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, mmn,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('r_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # metric_logger.add_meter('s_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # metric_logger.add_meter('t_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_c(optimizer, data_iter_step / len(data_loader) + epoch, args)

        len_num = 0
        if args.len_closeness != 0:
            len_num += 1
        if args.len_period != 0:
            len_num += 1
        if args.len_trend != 0:
            len_num += 1
        if len_num == 1:
            flows_xc, flows_x, flows_y, _ = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_x = flows_x.to(device, non_blocking=True)
            flows_y = flows_y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss1, all_loss1, poi_loss, s_loss, t_loss, sd_loss, td_loss, _, _ = model(flows_xc, flows_x, flows_y)

        elif len_num == 2:
            flows_xc, flows_xp, flows_x, flows_y, _ = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_xp = flows_xp.to(device, non_blocking=True)
            flows_x = flows_x.to(device, non_blocking=True)
            flows_y = flows_y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss1, all_loss1, poi_loss, s_loss, t_loss, sd_loss, td_loss, _, _ = model(flows_xc, flows_xp, flows_x, flows_y)
        else:
            flows_xc, flows_xp, flows_xt, flows_x, flows_y, _ = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_xp = flows_xp.to(device, non_blocking=True)
            flows_xt = flows_xt.to(device, non_blocking=True)
            flows_x = flows_x.to(device, non_blocking=True)
            flows_y = flows_y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss1, all_loss1, poi_loss, s_loss, t_loss, sd_loss, td_loss, _, _ = model(flows_xc, flows_xp, flows_xt, flows_x, flows_y)

        t = 1
        # if epoch <= 100:
        #     t = 0.01
        # else:
        #     # t = 0.0001 * math.pow(10, epoch // 100)
        #     t = 1

        loss = t * all_loss1 + args.alpha * s_loss + args.beta * t_loss + args.d_alpha * sd_loss + args.d_beta * td_loss + args.gama * poi_loss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        loss_value = math.sqrt(loss_value)
        r_loss_value = math.sqrt(loss1.item())
        s_loss_value = math.sqrt(s_loss.item())
        sd_loss_value = math.sqrt(sd_loss.item())
        t_loss_value = math.sqrt(t_loss.item())
        td_loss_value = math.sqrt(td_loss.item())
        poi_loss_value = poi_loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(r_loss=r_loss_value)
        metric_logger.update(poi_loss=poi_loss_value)
        metric_logger.update(s_loss=s_loss_value)
        metric_logger.update(sd_loss=sd_loss_value)
        metric_logger.update(t_loss=t_loss_value)
        metric_logger.update(td_loss=td_loss_value)


        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            # epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch)
            log_writer.add_scalar('lr', lr, epoch)
            log_writer.add_scalar('r_loss', r_loss_value, epoch)
            log_writer.add_scalar('poi_loss', poi_loss_value, epoch)
            log_writer.add_scalar('s_loss', s_loss_value, epoch)
            log_writer.add_scalar('sd_loss', sd_loss_value, epoch)
            log_writer.add_scalar('t_loss', t_loss_value, epoch)
            log_writer.add_scalar('td_loss', td_loss_value, epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

