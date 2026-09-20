# -*- coding: utf-8 -*-
"""Fill GDUPT grad innovation project application form (TARGET_DOCX copy)."""
import docx

PATH = r'work\apply\申请书_TARGET.docx'
d = docx.Document(PATH)

def append_to_para(p, text):
    """Append text to paragraph end, keeping existing runs."""
    p.add_run(text)

def set_cell(cell, text):
    """Replace cell text, keep first run's formatting if present."""
    paras = cell.paragraphs
    if not paras:
        cell.add_paragraph(text)
        return
    p0 = paras[0]
    if p0.runs:
        p0.runs[0].text = text
        for r in p0.runs[1:]:
            r.text = ''
    else:
        p0.add_run(text)
    for extra in paras[1:]:
        extra.text = ''

# ---------- 封面段落 ----------
paras = d.paragraphs
proj_name = '面向多尺度认知记忆动力学的学生知识状态智能追踪模型研究'
for i, p in enumerate(paras):
    t = p.text.strip()
    if t.startswith('项  目  名  称'):
        append_to_para(p, proj_name)
    elif t.startswith('培  养  学  院'):
        append_to_para(p, '（待填：所在学院）')
    elif t.startswith('项 目 负 责 人'):
        append_to_para(p, '（待填）')
    elif t.startswith('负 责 人 学 号'):
        append_to_para(p, '（待填）')
    elif t.startswith('专  业  名  称'):
        append_to_para(p, '（待填）')
    elif t.startswith('申  请  时  间'):
        append_to_para(p, '2026年    月    日')

# ---------- 表0 基本情况 ----------
t0 = d.tables[0]
# R0: 项目名称（合并cell，取第3列）
set_cell(t0.rows[0].cells[3], proj_name)
# R1: 起止时间（合并cell 4-9）
set_cell(t0.rows[1].cells[3], '2027年3月 至 2028年3月')
# R2: 姓名/性别/学号/出生年月
set_cell(t0.rows[2].cells[2], '（待填）')
set_cell(t0.rows[2].cells[4], '（待填）')
set_cell(t0.rows[2].cells[6], '（待填）')
set_cell(t0.rows[2].cells[9], '（待填）')
# R3: 学院/专业/层次（层次已填硕士研究生）
set_cell(t0.rows[3].cells[2], '（待填）')
set_cell(t0.rows[3].cells[5], '（待填）')
# R4: 导师/联系电话/电子信箱
set_cell(t0.rows[4].cells[2], '（待填）')
set_cell(t0.rows[4].cells[6], '（待填）')
set_cell(t0.rows[4].cells[9], '（待填）')
# R6: 负责人行（序号1）
set_cell(t0.rows[6].cells[1], '1')
set_cell(t0.rows[6].cells[2], '（待填：项目负责人）')
set_cell(t0.rows[6].cells[4], '（待填）')
set_cell(t0.rows[6].cells[6], '（待填）')
set_cell(t0.rows[6].cells[8], '模型设计与实验分析、论文撰写')
# R7-R10 组员2-5 留空

# ---------- 表1 研究基础（≤800字） ----------
base = (
    '本项目已在前期完成核心模型的自主研发，具备扎实的研究基础。'
    '（1）模型构建：完成知识追踪模型A2G的设计与实现，提出概念证据、双通道状态空间记忆与注意力检索'
    '三模态联合建模框架，实现概念边界分割的状态记忆机制与近因加权的遗忘感知证据融合。'
    '（2）实验验证：在ASSIST2017公开数据集上完成全流程训练与评估，验证集AUC达到0.8011、'
    '准确率0.7392，已超过同协议下多类基线方法。'
    '（3）效率优化：完成共享嵌入压缩，模型参数量由510.8万降至429.4万（降低约16%）且性能无损。'
    '（4）工程基础：建立完整的可复现实验体系，包括代码冻结副本、数据文件哈希校验、统一训练协议，'
    '单元测试104项全部通过，实验可复现、可审计。'
    '（5）在研工作：消融实验批次正在运行，将系统验证各创新模块的独立贡献，为论文提供完整证据链。'
    '（6）软硬件条件：具备GPU训练环境，掌握PyTorch深度学习框架与完整实验流程，具备独立完成'
    '本课题研究的能力。'
)
set_cell(d.tables[1].rows[0].cells[0], base)

# ---------- 表2 项目论证 ----------
t2 = d.tables[2]
# R0 立项依据（≤1000字）
lixiang = (
    '（一）项目来源与意义。知识追踪是智慧教育的核心基础问题，通过建模学生作答序列以实时估计其知识掌握状态，'
    '广泛支撑薄弱知识点诊断、个性化练习推荐与课程自适应学习。在人工智能赋能教育的国家战略背景下，'
    '构建高精度、可解释、高效的学生知识状态追踪模型具有重要的理论意义与应用价值。本项目属于人工智能与教育'
    '科学交叉方向，结合本专业深度学习研究方向提出。'
    '（二）国内外研究现状。早期工作以循环神经网络为主，如深度知识追踪（DKT，Piech等，NeurIPS 2015）；'
    '随后引入记忆网络（DKVMN，Zhang等，CIKM 2017）与自注意力机制（SAKT、AKT，Pandey等，EDM 2019），'
    '利用注意力权重增强可解释性。近年来，状态空间模型（SSM）以其线性复杂度的序列建模能力进入该领域'
    '（Mamba4KT，2024），并有工作将SSM与注意力混合以兼顾效率与记忆检索（ASIKT，SIGIR 2025）；'
    '亦有工作引入遗忘曲线建模（KVFKT，COLING 2025）。总体趋势是从单一序列模态走向多模态融合与认知理论结合。'
    '（三）现有不足与发展趋势。现有方法普遍存在三点不足：一是仅建模题目-作答序列，缺乏对概念级统计证据的'
    '显式利用；二是状态更新机制单一，未区分工作记忆与长时记忆的差异化演化；三是对遗忘过程的建模多为启发式。'
    '本项目拟针对上述不足，将概念证据、双通道状态记忆与注意力检索统一于认知记忆框架下建模，'
    '与现有工作形成明确差异。'
)
set_cell(t2.rows[0].cells[0], lixiang)

# R1 研究目标/方法/技术路线/创新性/可行性（≤1500字）
mubiao = (
    '（一）研究目标。构建面向多尺度认知记忆动力学的学生知识状态智能追踪模型，在保持高效性的前提下，'
    '提升知识状态预测精度与可解释性，并在多个公开数据集上验证其泛化能力。'
    '（二）拟解决的关键问题。①概念级证据如何调制记忆演化过程；②如何区分并联合建模工作记忆'
    '（快通道）与长时记忆（慢通道）的差异化更新；③如何在融合多模态信息的同时控制计算开销。'
    '（三）研究方法与技术路线。以深度学习序列建模为基础，采用"数据预处理—模型训练—消融验证—基线对比—'
    '跨数据集泛化—论文产出"的技术路线。模型以三模态联合建模为核心：概念证据流对历史作答进行概念级统计'
    '并以近因加权方式融合遗忘信息；双通道状态空间记忆模块通过概念边界分割实现工作记忆与长时记忆的差异化'
    '演化；注意力检索模块从历史记忆中检索与当前作答相关的信息用于预测。解码端融合认知诊断先验，'
    '输出学生掌握概率。'
    '（四）创新性。C1：提出概念证据、双通道状态记忆与注意力检索的三模态联合建模框架，现有方法未见'
    '同等设计；C2：提出概念边界分割的状态空间记忆机制，将状态通道按概念边界拆分，实现细粒度记忆演化；'
    'C3：提出近因加权的遗忘感知证据融合，显式耦合遗忘曲线；C4：提出共享嵌入压缩策略，在保证性能前提下'
    '显著降低模型参数规模。'
    '（五）可行性分析。①前期已完成模型实现并在ASSIST2017上取得AUC 0.8011的验证结果，技术路线已走通；'
    '②消融实验批次正在运行，实验体系完整可复现；③完成期限1年，进度安排合理；④指导教师研究方向一致，'
    '可提供条件保障。'
)
set_cell(t2.rows[1].cells[0], mubiao)

# R2 进度（≤600字）
jindu = (
    '第1—3个月：完成消融实验与基线方法同协议重训（在ASSIST2017、ASSIST2009等2—3个公开数据集上），'
    '验证各创新模块的独立贡献与模型泛化能力。'
    '第4—6个月：完成学术论文撰写，投稿至EI收录期刊或北大中文核心期刊（学校认定E类及以上论文），'
    '同步整理开源代码与实验复现说明。'
    '第7—9个月：根据审稿意见补充实验与修改论文，跟踪录用状态。'
    '第10—12个月：整理结题材料，提交结题报告，完成项目验收与成果标注。'
)
set_cell(t2.rows[2].cells[0], jindu)

# R3 预期成果（≤300字）
chengguo = (
    '（一）发表学校科研工作评价管理办法认定的E类及以上学术论文至少1篇（拟投EI收录期刊或北大中文核心期刊），'
    '第一作者为项目负责人，第一单位为广东石油化工学院，并标注"广东石油化工学院研究生科技创新计划项目"'
    '及项目批准号。'
    '（二）形成可复现的知识追踪模型开源代码与实验数据集处理流程。'
    '（三）模型在2—3个公开数据集上达到与最新方法相当或更优的预测精度，为结题验收提供成果支撑。'
)
set_cell(t2.rows[3].cells[0], chengguo)

# ---------- 表3 经费预算 ----------
t3 = d.tables[3]
budget = {
    1: ('1500', 'GPU计算资源与数据分析测试费用'),
    4: ('500', '购置深度学习与教育数据挖掘领域书籍、文献资料'),
    8: ('3000', '论文版面费及审稿费'),
}
for ri, (amt, note) in budget.items():
    set_cell(t3.rows[ri].cells[1], amt)
    set_cell(t3.rows[ri].cells[2], note)
set_cell(t3.rows[13].cells[1], '5000')

d.save(PATH)
print('filled OK')

# 字数统计
def wc(s): return len(s.replace(' ', ''))
print('研究基础字数:', wc(base))
print('立项依据字数:', wc(lixiang))
print('目标方法字数:', wc(mubiao))
print('进度字数:', wc(jindu))
print('预期成果字数:', wc(chengguo))
