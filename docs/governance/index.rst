.. -*- mode: rst -*-

Governance
==========

**Version 1.1 on May 15, 2019**

Purpose
-------
Yellowbrick has grown from a project with a handful of people hacking on it in their spare time into one with thousands of users, dozens of contributors, and several organizational affiliations. The burden of the effort to maintain the project has so far been borne by a few individuals dedicated to its success. However, as the project continues to grow, we believe it is important to establish a system that clearly defines how the project will be administered, organized, and maintained in the coming years. We hope that this clarity will lighten the administrative load of the maintainers and core-developers, streamline decision making, and allow new people to participate meaningfully. Most importantly, we hope that the product of these effects is to ensure the members of the Yellowbrick project are able to maintain a healthy and productive relationship with the project, allowing them to avoid burnout and fatigue.

The purpose of this document is to establish a system of governance that will define the interactions of Yellowbrick members and facilitate decision-making processes. This governance document serves as guidance to Yellowbrick members, and all Yellowbrick members must agree to adhere to its guidance in order to participate in the project. Although not legally binding, this document may also serve as bylaws for the organization of Yellowbrick contributors. We believe this document should be treated as a living document that is amended regularly in order to support the future objectives of the project.

Organization
------------
The Yellowbrick project, also referred to as scikit-yellowbrick or scikit-yb, is an open source Python library for machine learning visual analysis and diagnostics. The library is licensed by the Apache 2.0 open source license, the source code is version controlled on GitHub, and the package is distributed via PyPI. The project is maintained by volunteer contributors who refer to themselves corporately as “the scikit-yb developers”.

Founders and Leadership
~~~~~~~~~~~~~~~~~~~~~~~
Yellowbrick was founded by Benjamin Bengfort and Rebecca Bilbro, who were solely responsible for the initial prototypes and development of the library. In the tradition of Python, they are sometimes referred to as the “benevolent dictators for life”, or BDFLs, of the project. For the purpose of this document and to emphasize the role of maintainers and core-contributors, they are simply referred to as “the founders”. From a governance perspective, the founders have a special role in that they provide vision, thought leadership, and high-level direction for the project’s members and contributors.

The Yellowbrick project is bigger than two people, however, therefore the primary mechanism of governance will be a collaboration of contributors and advisors in the tradition of the Apache Software Foundation. This collaboration ensures that the most active project contributors and users (the primary stakeholders of the project) have a voice in decision-making responsibilities. Note that Yellowbrick defines who active project contributors are (e.g. advisors, maintainers, core-contributors, and contributors) in a very specific and time-limited fashion. Depending on the type of required action, a subset of the active project contributors will deliberate and make a decision through majority voting consensus, subject to a final say of the founders. To ensure this happens correctly, contributors must be individuals who represent themselves and not companies.

Contributors and Roles
~~~~~~~~~~~~~~~~~~~~~~
There are many ways to participate as a contributor to the Yellowbrick community, but first and foremost, all Yellowbrick contributors are users. Only by using Yellowbrick do contributors gain the requisite understanding and context to meaningfully contribute. Whether you use Yellowbrick as a student to learn machine learning (or as a teacher to teach it) or you are a professional data scientist or analyst who uses it for visual analytics or diagnostics, you cannot be a contributor without some use-case driven motivation. This definition also specifically excludes other motivations, such as financial motivations or attention-seeking. The reason for this is simple, Yellowbrick is a technical library that is intended for a technical audience.

You become a contributor when you give something back to the Yellowbrick community. A contribution can be anything large or small—related to code or something unique to your talents. In the end, the rest of the community of contributors will decide what constitutes a meaningful contribution, but some of the most common contributions include:

- Successfully merging a pull request after a code review.
- Contributing to the scikit-yb documentation.
- Translating the scikit-yb documentation into other languages.
- Submitting detailed, reproducible bug reports.
- Submitting detailed, prototyped visualizer or feature requests.
- Writing blog posts or giving talks that demonstrate Yellowbrick’s use.
- Answering questions on Stack Overflow or on our GitHub issues.
- Organizing a Yellowbrick users meetup or other events.

Once you have contributed to Yellowbrick, you *will always be a contributor*. We believe this is a badge of honor and we hold all Yellowbrick contributors in special esteem and respect.

If you are routinely or frequently contributing to Yellowbrick, you have the opportunity to assume one of the roles of an active project contributor. These roles allow you to fundamentally guide the course of the Yellowbrick project but are also time-limited to help us avoid burnout and fatigue. We would like to plan our efforts as fixed-length sprints rather than as an open-ended run.

Note that the roles described below are not mutually exclusive. A Yellowbrick contributor can simultaneously be a core-contributor, a maintainer, and an advisor if they would like to take on all of those responsibilities at the same time. None of these roles “outranks” another, they are simply a delineation of different responsibilities. In fact, they are designed to be fluid so that members can pick and choose how they participate at any given time. A detailed description of the roles follows.

Core Contributor
^^^^^^^^^^^^^^^^
A core-contributor commits to actively participate in a 4-month *semester* of Yellowbrick development, which culminates in the next release (we will discuss semesters in greater detail later in this document). At the beginning of the semester, the core-contributor will outline their participation and goals for the release, usually by accepting the assignment of 1-5 issues that should be completed before the semester is over. Core-contributors work with the maintainers over the course of the semester to ensure the library is at the highest quality. The responsibilities of core-contributors are as follows:

- Set specific and reasonable goals for contributions over the semester.
- Work with maintainers to complete issues and goals for the next release.
- Respond to maintainer requests and be generally available for discussion.
- Participating actively on the #yellowbrick Slack channel.
- Join any co-working or pair-programming sessions required to achieve goals.

Core-contributors can reasonably set as high or as low a challenge for themselves as they feel comfortable. We expect core-contributors to work on Yellowbrick for a total of about 5 hours per week. If core-contributors do not meet their goals by the end of the semester, they will likely not be invited to participate at that level in the next semester, though they may still volunteer subject to the approval of the maintainers for that semester.

Maintainer
^^^^^^^^^^
Maintainers commit to actively manage Yellowbrick during a 4-month semester of Yellowbrick development and are primarily responsible for building and deploying the next release. Maintainers may also act as core-contributors and use the dedicated semester group to ensure the release objectives set by the advisors are met. This primarily manifests itself in the maintenance of GitHub issues and code reviews of Pull Requests. The responsibilities of maintainers are as follows:

- Being active and responsive on the #yellowbrick and #oz-maintainers Slack channels.
- Being active and responsive on the team-oz/maintainers GitHub group.
- Respond to user messages on GitHub and the ListServ (Stack Overflow, etc).
- Code review pull requests and safely merge them into the project.
- Maintain the project’s high code quality and well-defined API.
- Release the next version of Yellowbrick.
- Find and share things of interest to the Yellowbrick community.

The maintainer role is an exhausting and time-consuming role. We expect maintainers to work on Yellowbrick 10-20 hours per week. Maintainers should have first been core-contributors so they understand what the role entails (and to allow a current maintainer to mentor them to assume the role). Moreover, we recommend that maintainers periodically/regularly take semesters off to ensure they don’t get burnt out.

Coordinator
^^^^^^^^^^^
If a semester has a large enough group of maintainers and core-contributors, we may optionally appoint a contributor as a coordinator. Coordinators specialize in the project management aspects of a version release and also direct traffic to the actively participating group. The coordinator may or may not be a maintainer. If the coordinator is nominated from the maintainers, the coordination role is an auxiliary responsibility. Otherwise, the coordinator may be a dedicated contributor that  focuses only on communication and progress. The coordinator’s primary responsibilities are as follows:

- Ensure that the features, bugs, and issues assigned to the version release are - progressing.
- Be the first to respond welcomingly to new contributors.
- Assign incoming pull requests, issues, and other responses to maintainers.
- Communicate progress to the advisors and to the Yellowbrick community.
- Encourage and motivate the active contributors group.
- Coach core-contributors and maintainers.
- Organize meetups, video calls, and other social and engagement activities.

The coordinator’s role is mostly about communication and the amount of dedicated effort can vary depending on external factors. Based on our experience it could be as little as a couple of hours per week to as much work as being a maintainer. The coordinator’s responsibilities are organizational and are ideal for someone who wants to be a part of the community but has less of a software background. In fact, we see this role as being similar to a tech project management role, which is often entry level and serves as experience to propel coordinators into more active development. Alternatively, the coordinator can be a more experienced maintainer who needs a break from review but is willing to focus on coaching.

Mentor-Contributor
^^^^^^^^^^^^^^^^^^

A mentor-contributor assumes all the responsibilities of a core contributor but commits 25-50% of their time toward mentoring a new contributor who has little-to-no experience. The purpose of the mentor-contributor role is to assist in recruiting and developing a pipeline of new contributors for Yellowbrick.

A mentor-contributor would guide a new contributor in understanding the community, roles, ethos, expectations, systems, and processes in participating and contributing to an open source project such as Yellowbrick. A new contributor is someone who is interested in open source software projects but has little-to-no experience working on an open source project or participating in the community.

The mentor-contributor would work in tandem with the new contributor during the semester to complete assigned issues and prepare the new contributor to be a core contributor in a future semester. A mentor-contributor would mostly likely work on fewer issues during the semester in order to allocate time for mentoring. Mentor-contributors would be matched with new contributors prior to the start of each semester or recruit their own contributor (eg. colleague or friend). The responsibilities of mentor-contributors are as follows:

- When possible, recruit new contributors to Yellowbrick.
- Schedule weekly mentoring sessions and assign discrete yet increasingly complex tasks to the new contributor.
- Work with the new contributor to complete assigned issues during the semester.
- Set specific and reasonable goals for contributions over the semester.
- Work with maintainers to complete issues and goals for the next release.
- Respond to maintainer requests and be generally available for discussion.
- Participating actively on the #yellowbrick Slack channel.
- Join any co-working or pair-programming sessions required to achieve goals.
- Make a determination at the end of the semester on the readiness of the new contributor to be a core-contributor.

The mentor-contributor role may also appeal to students who use Yellowbrick in a machine learning course and seek the experience of contributing to an open source project.

Advisor
^^^^^^^
Advisors are the primary decision-making body for Yellowbrick. They serve over the course of 3 semesters (1 calendar year) and are broadly responsible for defining the roadmap for Yellowbrick development for the next three releases. Advisors meet regularly, at least at the beginning of every semester, to deliberate and discuss next steps for Yellowbrick and to give guidance for core-contributors and maintainers for the semester. The detailed responsibilities of the advisors are as follows:

- Contribute dues based on the number of advisors in a year to meet fixed running costs.
- Make decisions that affect the entire group through votes (ensure a quorum).
- Meet at least three times a year at the beginning of each semester (or more if required).
- Elect officers to conduct the business of Yellowbrick.
- Ensure Yellowbrick’s financial responsibilities are met.
- Create a roadmap for Yellowbrick development.
- Approve the release of the next Yellowbrick version.
- Approve core-contributor and maintainer volunteers.
- Ensure Yellowbrick is growing by getting more users.
- Ensure Yellowbrick is a good citizen of the PyData and Python communities.
- Recruit more contributors to participate in the Yellowbrick community.

The advisor role is primarily intended to allow members to guide the course of Yellowbrick and actively participate in decisions without making them feel that they need to participate as maintainers (seriously, take a break - don’t burn out!). As such, the role of advisors is limited to preparing for and participating in the semester meetings or any other meetings that are called. Therefore, we assume that the time commitment of an advisor is roughly 30-40 hours per year (less than 1 hour per week).

The board of advisors is open to every contributor, and in fact, is open to all users because joining the board of advisors requires a financial contribution. Yellowbrick has limited fixed running costs every year, for example, $10 a month to Read The Docs and $14.99 for our domain name. When you volunteer to be an advisor for a year, you commit to paying an equal portion of those fixed running costs, based on the number of advisor volunteers. This also ensures that we have a fixed quorum to ensure votes run smoothly.

Affiliations
~~~~~~~~~~~~
Yellowbrick may be affiliated with one or more organizations that promote Yellowbrick and provide financial assistance or other support. Yellowbrick affiliations are determined by the advisors who should work to ensure that affiliations are in both the organization’s and Yellowbrick’s best interests. Yellowbrick is currently affiliated with:

- District Data Labs
- NumFOCUS
- Georgetown University

The advisors should update this section to reflect all current and past affiliations.

Operations
----------
The primary activity of Yellowbrick contributors is to develop the Yellowbrick library into the best tool for visual model diagnostics and machine learning analytics. A secondary activity is to support Yellowbrick users and to provide a high level of customer service. Tertiary activities include being good citizens of the Python and PyData communities and to the open source community at large. These activities are directed to the primary mission of Yellowbrick: to greatly enhance the machine learning workflow with visual steering and diagnostics.

Although Yellowbrick is not a commercial enterprise, it does not preclude its members from pursuing commercial activities using Yellowbrick subject to our license agreement.

In this section, we break down the maintenance and development operations of the project. These operations are a separate activity from administration and decision making, which we will discuss in the following section.

Semesters and Service Terms
~~~~~~~~~~~~~~~~~~~~~~~~~~~
In order to ensure that maintainers are able to contribute meaningfully and effectively without getting burned out, we divided the maintenance and development life cycle into three semesters a year as follows:

- Spring semester: January - April
- Summer semester: May - August
- Fall semester: September - December

Every maintainer and core-contributor serves in these terms and is only expected to participate at the commitment level described by the roles section for the term they have joined. At the end of the term, there is no expectation that the maintainer must continue to the next term. If they wish to do so, they must volunteer to participate in the next term. We hope that this allows maintainers to be more strategic in their participation, e.g. participating every other semester, or taking off a semester where they know they will be busy with other obligations.

An advisor’s service term is 1 year, and they must accept the advisory role by the end of January in that calendar year by paying the dues computed by the number of participating advisors. Advisors can remain advisors so long as they wish to participate by paying the dues, though if advising meetings are unable to achieve quorum; absent advisors may be asked to step down.

The goal of the advising service term is to allow maintainers who wish to take a semester off to still have a meaningful say in the development of the project. We hope that this will also facilitate maintainers feeling that they can take a break without being cut out of the loop, and allowing them to rejoin the project as maintainers or core-contributors in a meaningful way when they are ready again.

Release Guidelines
~~~~~~~~~~~~~~~~~~
The focus of the semester is to release the next version of Yellowbrick. The advisors set a roadmap based on the current issues and features they wish to address. Core-contributors volunteer to take on specific issues and maintainers are responsible for organizing themselves and the core-contributors to complete the work in the version, releasing it at the end of the semester.

Maintainers may also include in the semester release any other contributions or pull requests made by members of the community at their discretion. Maintainers should address all pull-requests and opened issues (as well as emails on the listserv and questions on Stack Overflow, Twitter, etc.) - facilitating their inclusion in the release, or their closure if they become stale or are out of band.

During the middle of a semester, maintainers may be required to issue a hotfix release for time-critical or security-related fixes. Hotfix releases should be small, incremental improvements on the previous release. We hope that the expected release schedule of April, August, and December also assists the user community in giving feedback and guidance about the direction of Yellowbrick.

Advisors
--------
Yellowbrick advisors are contributors who take on the responsibility of setting the vision and direction for the development of the project. They may, for example, make decisions about which features are focused on during a semester or to apply for a small development grant and use the funds to improve Yellowbrick. They may also ask to affiliate with other projects and programs that are tangential to but crucial to the success of Yellowbrick - e.g. organizing workshops or talks, creating t-shirts or stickers, etc.

Advisors influence Yellowbrick primarily through advisor meetings. This section describes advisor interactions, meetings, and decision-making structure.

Officers
~~~~~~~~
During the first advisor meeting in January, three officers should be elected to serve in special capacities over the course of the year. The officer positions are as follows:

**Chair**: the chair’s responsibility is to organize and lead the advisor meetings, to ensure good conduct and order, and to adjudicate any disputes. The chair may call additional advisor meetings and introduce proposals for the advisors to make decisions upon. The chair calls votes to be held and may make a tie-breaking vote if a tie exists. At the beginning of every meeting, the chair should report the status of Yellowbrick and the state of the current release.

**Secretary**: the secretary’s responsibility is to create an agenda and record the minutes of advisor meetings, including all decisions undertaken and their results. The secretary may take the place of the chair if the chair is not available during a meeting.

**Treasurer**: the treasurer assumes responsibility for tracking cash flow—both dues payments from the advisors as well as any outgoing payments for expenses. The treasurer may also be responsible to handle monies received from outside sources, such as development grants. The treasurer should report on the financial status of the project at the beginning of every advisor meeting.

Officers may either self-nominate or be nominated by other advisors. Nominations should be submitted before the first advisor meeting in January, and the first order of business for that meeting should be to vote the officers into their positions. Officer votes follow the normal rules of voting as described below; in the case of a tie, the founders cast the tie-breaking decision.

Meetings
~~~~~~~~
Advisors must schedule at least three meetings per year in the first month of each semester. At the advisor’s discretion, they may schedule additional meetings as necessary. No vote may take place without a meeting and verbal deliberation (e.g. no voting by email or by poll). Meetings must be held with a virtual component. E.g. even if the meeting is in-person with available advisors, there must be some teleconference or video capability to allow remote advisors to participate meaningfully in the meetings.

Scheduling
^^^^^^^^^^
To schedule a meeting, the chair must prepare a poll to find the availability of all advisors with at least 6 options over the next 10 days. The chair must ensure that every reasonable step is taken to include as many of the advisors as possible. No meeting may be scheduled without a quorum attending.

Meetings must be scheduled in January, May, and September; as close to the start of the semester as possible. It is advisable to send the scheduling poll for those meetings in December, April, and August. Advisors may hold any number of additional meetings.

Voting
^^^^^^
Voting rules are simple—a proposal for a vote must be made by one of the advisors, submitted to the chair who determines if a vote may be held. The proposal may be seconded to require the chair to hold a vote on the issue. Votes may only be held if a quorum of all advisors (half of the advisors plus one) is present or accounted for in some way (e.g. through a delegation). Prior to the vote, a period of discussion of no less than 5 minutes must be allowed so that members may speak for or against the proposal.

Votes may take place in two ways, the mechanism of which must be decided by the chair as part of the vote proposal. The vote can either be performed by secret ballot as needed or, more generally, by counting the individual votes of advisors. Advisors can either submit a “yea”, “nay”, or “abstain” vote. A proposal is passed if a majority (half of the advisors plus one) submit a “yea” vote. A proposal is rejected if a majority (half of the advisors plus one) submit a “nay” vote. If neither a majority “yea” or “nay” votes are obtained, the vote can be conducted again at the next advisors meeting.

The proposal is not ratified until it is agreed upon by the founders, who have the final say in all decision making. This can be interpreted as either a veto (the founders reject a passed proposal) or as a request for clarification (the founders require another vote for a rejected proposal). There is no way to overturn a veto by the founders, though they recognize that by “taking their ball and going home”, they are not fostering a sense of community and expect that these disagreements will be extraordinarily rare.

Agenda and Minutes
^^^^^^^^^^^^^^^^^^
The secretary is responsible for preparing a meeting agenda and sending it to all advisors at least 24 hours before the advisors meeting. Minutes that describe in detail the discussion, any proposals, and the outcome of all votes should be taken as well. Minutes should be made public to the rest of the Yellowbrick community.

Amendments
----------
Amendments or changes to this governance document can be made by a vote of the advisors. Proposed changes must be submitted to the advisors for review at least one week prior to the vote taking place. Prior to the vote, the advisors should allow for a period of discussion where the changes or amendments can be reviewed by the group. Amendments are ratified by the normal voting procedure described above.

Amendments should update the version and date at the top of this document.

Board of Advisors Minutes
-------------------------

.. toctree::
    :maxdepth: 1

    minutes/2019-05-15.rst
    minutes/2019-09-09.rst
    minutes/2020-01-07.rst
    minutes/2020-05-13.rst
    minutes/2020-10-06.rst
